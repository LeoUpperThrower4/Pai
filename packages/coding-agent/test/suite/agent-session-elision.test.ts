import { mkdirSync, mkdtempSync, readdirSync, readFileSync, renameSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fauxAssistantMessage, fauxToolCall } from "@earendil-works/pi-ai";
import type { ToolResultMessage } from "@earendil-works/pi-ai/compat";
import { afterEach, describe, expect, it, vi } from "vitest";
import { SessionManager } from "../../src/core/session-manager.ts";
import { BUILTIN_SLASH_COMMANDS } from "../../src/core/slash-commands.ts";
import { InteractiveMode } from "../../src/modes/interactive/interactive-mode.ts";
import { createHarness, getMessageText, type Harness } from "./harness.ts";

function appendResult(manager: SessionManager, text: string, isError = false): string {
	return manager.appendMessage({
		role: "toolResult",
		toolCallId: `call-${manager.getEntryCount()}`,
		toolName: "read",
		content: [{ type: "text", text }],
		details: { original: text },
		isError,
		timestamp: Date.now(),
	});
}

function resultAt(manager: SessionManager, id: string): ToolResultMessage {
	const entry = manager.getEntry(id);
	if (entry?.type !== "message" || entry.message.role !== "toolResult") throw new Error("Missing tool result");
	return entry.message;
}

describe("manual tool-result elision", () => {
	const harnesses: Harness[] = [];
	afterEach(() => {
		while (harnesses.length) harnesses.pop()?.cleanup();
	});

	async function setup(manager = SessionManager.inMemory()): Promise<Harness> {
		const harness = await createHarness({
			sessionManager: manager,
			settings: { compaction: { enabled: false }, retry: { enabled: false } },
		});
		harnesses.push(harness);
		return harness;
	}

	it("summarizes every active tool response independently and preserves other messages and result identities", async () => {
		const harness = await setup();
		const manager = harness.sessionManager;
		const user = manager.appendMessage({ role: "user", content: "continue task", timestamp: Date.now() });
		const first = appendResult(manager, "long output ".repeat(100));
		const second = appendResult(manager, "ENOENT src/missing.ts", true);
		const originalFirst = resultAt(manager, first);
		harness.setResponses([
			(context) => {
				const text = context.messages.map(getMessageText).join("\n");
				expect(text).toContain("long output");
				expect(text).not.toContain("continue task");
				expect(text).not.toContain("ENOENT");
				return fauxAssistantMessage("Read src/large.ts containing the parser.");
			},
			(context) => {
				expect(context.messages.map(getMessageText).join("\n")).toContain("Failed: true");
				return fauxAssistantMessage("Read failed: ENOENT src/missing.ts.");
			},
		]);
		const userEntry = manager.getEntry(user);
		expect(await harness.session.elide()).toEqual({ count: 2 });
		expect(manager.getEntry(user)).toBe(userEntry);
		expect(resultAt(manager, first)).toMatchObject({ toolCallId: originalFirst.toolCallId, toolName: "read" });
		expect(resultAt(manager, first).details).toBeUndefined();
		expect(getMessageText(resultAt(manager, first))).toContain("Read src/large.ts");
		expect(resultAt(manager, second).isError).toBe(true);
		expect(harness.session.messages).toEqual(manager.buildSessionProjection().messages);
		expect(manager.getEntries().filter((entry) => entry.type === "usage" && entry.kind === "elision")).toHaveLength(
			2,
		);
		expect(await harness.session.elide()).toEqual({ count: 0 });
		expect(harness.getPendingResponseCount()).toBe(0);
	});

	it("invalidates old token estimates and sends summaries with intact call/result pairing on the next turn", async () => {
		const harness = await setup();
		const manager = harness.sessionManager;
		manager.appendMessage({ role: "user", content: "read a file", timestamp: Date.now() });
		const assistant = fauxAssistantMessage([fauxToolCall("read", { path: "src/a.ts" }, { id: "call-2" })], {
			stopReason: "toolUse",
		});
		manager.appendMessage(assistant);
		const id = appendResult(manager, "original output ".repeat(100));
		const completed = fauxAssistantMessage("Finished reading");
		completed.usage = { ...completed.usage, input: 100000, totalTokens: 100000 };
		manager.appendMessage(completed);
		harness.session.refreshContext();
		expect(harness.session.getContextUsage()?.tokens).toBe(100000);
		harness.setResponses([fauxAssistantMessage("File contains a parser.")]);
		await harness.session.elide();
		expect(harness.session.getContextUsage()?.tokens).toBeLessThan(1000);
		expect(manager.getEntries().some((entry) => entry.type === "message" && entry.message === assistant)).toBe(true);
		expect(manager.getEntries().some((entry) => entry.type === "message" && entry.message === completed)).toBe(true);
		harness.setResponses([
			(context) => {
				const result = context.messages.find((message) => message.role === "toolResult");
				expect(result).toMatchObject({ toolCallId: "call-2", toolName: "read" });
				expect(getMessageText(result)).toContain("File contains a parser.");
				expect(context.messages.map(getMessageText).join("\n")).not.toContain("original output");
				return fauxAssistantMessage("Next response");
			},
		]);
		await harness.session.prompt("continue");
		expect(getMessageText(resultAt(manager, id))).toContain("File contains a parser.");
	});

	it("does not overwrite the session if its branch changes while a summary is generated", async () => {
		const harness = await setup();
		const id = appendResult(harness.sessionManager, "original");
		harness.setResponses([
			() => {
				harness.sessionManager.resetLeaf();
				return fauxAssistantMessage("description");
			},
		]);
		await expect(harness.session.elide()).rejects.toThrow("Session changed");
		expect(getMessageText(resultAt(harness.sessionManager, id))).toBe("original");
	});

	it("only elides visible results after compaction on the selected branch", async () => {
		const harness = await setup();
		const manager = harness.sessionManager;
		const old = appendResult(manager, "compacted output");
		const retained = appendResult(manager, "retained output");
		manager.appendCompaction("checkpoint", retained, 100);
		const omitted = appendResult(manager, "omitted output");
		manager.appendContextEdit(omitted, null);
		const leaf = manager.getLeafId()!;
		const other = appendResult(manager, "other branch output");
		manager.branch(leaf);
		harness.setResponses([fauxAssistantMessage("Retained description")]);
		expect(await harness.session.elide()).toEqual({ count: 1 });
		expect(getMessageText(resultAt(manager, old))).toBe("compacted output");
		expect(getMessageText(resultAt(manager, omitted))).toBe("omitted output");
		expect(getMessageText(resultAt(manager, other))).toBe("other branch output");
		expect(
			manager
				.buildSessionProjection()
				.messages.some((message) => getMessageText(message).includes("Retained description")),
		).toBe(true);
	});

	it("removes original payloads and previous context-edit copies from disk across resume", async () => {
		const directory = mkdtempSync(join(tmpdir(), "pi-elision-"));
		try {
			const manager = SessionManager.create(directory, directory);
			manager.appendMessage({ role: "user", content: "task", timestamp: Date.now() });
			const id = appendResult(manager, "RAW-ORIGINAL");
			manager.appendContextEdit(id, { content: "OLD-EDIT" });
			manager.appendContextEdit(id, { content: "VISIBLE-EDIT" });
			const harness = await setup(manager);
			harness.setResponses([
				(context) => {
					const text = context.messages.map(getMessageText).join("\n");
					expect(text).toContain("VISIBLE-EDIT");
					expect(text).not.toContain("RAW-ORIGINAL");
					return fauxAssistantMessage("Description only");
				},
			]);
			await harness.session.elide();
			const file = manager.getSessionFile()!;
			const contents = readFileSync(file, "utf8");
			for (const original of ["RAW-ORIGINAL", "OLD-EDIT", "VISIBLE-EDIT"]) expect(contents).not.toContain(original);
			const reopened = SessionManager.open(file);
			expect(getMessageText(resultAt(reopened, id))).toContain("Description only");
			expect(reopened.getEntry(id)).toMatchObject({ elided: true });
			expect(reopened.buildSessionProjection().messages).toEqual(manager.buildSessionProjection().messages);
		} finally {
			rmSync(directory, { recursive: true, force: true });
		}
	});

	it.each([
		fauxAssistantMessage("", { stopReason: "error", errorMessage: "bad request" }),
		fauxAssistantMessage("incomplete", { stopReason: "length" }),
		fauxAssistantMessage("", { stopReason: "aborted" }),
		fauxAssistantMessage("   "),
	])("does not replace any results when a summary fails: %j", async (failure) => {
		const harness = await setup();
		const first = appendResult(harness.sessionManager, "first original");
		const second = appendResult(harness.sessionManager, "second original");
		harness.setResponses([fauxAssistantMessage("first summary"), failure]);
		await expect(harness.session.elide()).rejects.toThrow();
		expect(getMessageText(resultAt(harness.sessionManager, first))).toBe("first original");
		expect(getMessageText(resultAt(harness.sessionManager, second))).toBe("second original");
		expect(harness.session.isIdle).toBe(true);
	});

	it("supports cancellation and rejects concurrent elision and prompting", async () => {
		const harness = await setup();
		const id = appendResult(harness.sessionManager, "original");
		let entered!: () => void;
		const started = new Promise<void>((resolve) => {
			entered = resolve;
		});
		let finish!: () => void;
		const waiting = new Promise<void>((resolve) => {
			finish = resolve;
		});
		harness.setResponses([
			async () => {
				entered();
				await waiting;
				return fauxAssistantMessage("description");
			},
		]);
		const operation = harness.session.elide();
		await started;
		expect(harness.session.isIdle).toBe(false);
		await expect(harness.session.elide()).rejects.toThrow("Wait for");
		await expect(harness.session.prompt("new task")).rejects.toThrow("compaction");
		await expect(
			harness.session.sendCustomMessage(
				{ customType: "test", content: "new task", display: false },
				{ triggerTurn: true },
			),
		).rejects.toThrow("elision");
		harness.session.abortCompaction();
		finish();
		await expect(operation).rejects.toThrow();
		expect(getMessageText(resultAt(harness.sessionManager, id))).toBe("original");
		expect(harness.session.isIdle).toBe(true);
	});

	it("rejects generated tool calls instead of executing or persisting them", async () => {
		const harness = await setup();
		const id = appendResult(harness.sessionManager, "original");
		harness.setResponses([
			fauxAssistantMessage([fauxToolCall("bash", { command: "echo unsafe" })], { stopReason: "toolUse" }),
		]);
		await expect(harness.session.elide()).rejects.toThrow("attempted to call a tool");
		expect(getMessageText(resultAt(harness.sessionManager, id))).toBe("original");
	});

	it("forwards images to the summarizer and permanently removes their data", async () => {
		const harness = await setup();
		harness.session.agent.state.model = { ...harness.getModel(), input: ["text", "image"] };
		const image = { type: "image" as const, data: "image-bytes", mimeType: "image/png" };
		const id = harness.sessionManager.appendMessage({
			role: "toolResult",
			toolCallId: "image-call",
			toolName: "read",
			content: [image],
			isError: false,
			timestamp: Date.now(),
		});
		harness.setResponses([
			(context) => {
				const input = context.messages.find((message) => message.role === "user");
				expect(input?.content).toContainEqual(image);
				return fauxAssistantMessage("A screenshot showing a failed test.");
			},
		]);
		await harness.session.elide();
		expect(JSON.stringify(harness.sessionManager.getEntries())).not.toContain("image-bytes");
		expect(getMessageText(resultAt(harness.sessionManager, id))).toContain("screenshot");
	});

	it("does not silently discard images when the selected model cannot read them", async () => {
		const harness = await setup();
		harness.session.agent.state.model = { ...harness.getModel(), input: ["text"] };
		harness.sessionManager.appendMessage({
			role: "toolResult",
			toolCallId: "image-call",
			toolName: "read",
			content: [{ type: "image", data: "image-bytes", mimeType: "image/png" }],
			isError: false,
			timestamp: Date.now(),
		});
		await expect(harness.session.elide()).rejects.toThrow("image-capable model");
	});

	it("leaves in-memory results intact and cleans up the temporary file when persistence fails", () => {
		const directory = mkdtempSync(join(tmpdir(), "pi-elision-write-"));
		try {
			const manager = SessionManager.create(directory, directory);
			manager.appendMessage({ role: "user", content: "task", timestamp: Date.now() });
			const id = appendResult(manager, "original");
			const file = manager.getSessionFile()!;
			const savedFile = join(directory, "original.jsonl");
			renameSync(file, savedFile);
			mkdirSync(file);
			const originalLeaf = manager.getLeafId();
			expect(() => manager.elideToolResults(new Map([[id, "description"]]))).toThrow();
			expect(getMessageText(resultAt(manager, id))).toBe("original");
			expect(manager.getLeafId()).toBe(originalLeaf);
			expect(readFileSync(savedFile, "utf8")).not.toContain("description");
			expect(readdirSync(directory).some((name) => name.endsWith(".tmp"))).toBe(false);
		} finally {
			rmSync(directory, { recursive: true, force: true });
		}
	});

	it("returns without a model request when there are no tool results", async () => {
		const harness = await setup();
		expect(await harness.session.elide()).toEqual({ count: 0 });
	});

	it("routes /elision to the manual command handler instead of sending it as a prompt", async () => {
		const context = {
			defaultEditor: {} as { onSubmit?: (text: string) => Promise<void> },
			editor: { setText: vi.fn() },
			handleElisionCommand: vi.fn(async () => {}),
		};
		const prototype = InteractiveMode.prototype as unknown as {
			setupEditorSubmitHandler(this: typeof context): void;
		};
		prototype.setupEditorSubmitHandler.call(context);
		await context.defaultEditor.onSubmit?.("/elision");
		expect(context.handleElisionCommand).toHaveBeenCalledOnce();
		expect(context.editor.setText).toHaveBeenCalledWith("");
		expect(BUILTIN_SLASH_COMMANDS.some((command) => command.name === "elision")).toBe(true);
	});
});
