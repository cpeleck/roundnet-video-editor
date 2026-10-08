import { expect, test } from 'vitest';
import { createCommandQueue } from './commandQueue';

test('edits survive a running position save and use successive revisions', async () => {
  let release!: () => void;
  let revision = 1;
  const calls: { type: string; revision: number }[] = [];
  const queue = createCommandQueue(async type => {
    calls.push({ type, revision });
    if (calls.length === 1) await new Promise<void>(r => { release = r; });
    return ++revision;
  });
  const position = queue('position', { position: 1 }, 'p');
  const stale = queue('position', { position: 2 }, 'p');
  const latest = queue('position', { position: 3 }, 'p');
  expect(await stale).toBeUndefined();
  const edit = queue('add', { start_time: 2, end_time: 4 }, 'p');
  expect(await latest).toBeUndefined();
  release();
  expect(await position).toBe(2);
  expect(await edit).toBe(3);
  expect(calls).toEqual([{ type: 'position', revision: 1 }, { type: 'add', revision: 2 }]);
});

test('a failed edit does not stall the next edit', async () => {
  const queue = createCommandQueue(async type => { if (type === 'fail') throw new Error('failure'); return type; });
  expect(await queue('fail', {}, 'p')).toBeUndefined();
  expect(await queue('classify', {}, 'p')).toBe('classify');
});
