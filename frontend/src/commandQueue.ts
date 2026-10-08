/** Edits have priority over coalesced playback saves; each uses current state. */
export function createCommandQueue<T>(run: (type: string, values: Record<string, unknown>, projectId: string) => Promise<T | undefined>) {
  type Item = { type: string; values: Record<string, unknown>; projectId: string; resolve: (value: T | undefined) => void };
  const edits: Item[] = [];
  let position: Item | undefined;
  let running = false;
  const drain = async () => {
    if (running) return;
    running = true;
    try {
      while (edits.length || position) {
        const item = edits.shift() ?? position!;
        if (item === position) position = undefined;
        try { item.resolve(await run(item.type, item.values, item.projectId)); }
        catch { item.resolve(undefined); }
      }
    } finally { running = false; }
  };
  return (type: string, values: Record<string, unknown>, projectId: string) => new Promise<T | undefined>(resolve => {
    const item = { type, values, projectId, resolve };
    if (type === 'position') { position?.resolve(undefined); position = item; }
    else {
      // A pending old playhead must never undo an edit's deliberate seek.
      position?.resolve(undefined); position = undefined; edits.push(item);
    }
    void drain();
  });
}
