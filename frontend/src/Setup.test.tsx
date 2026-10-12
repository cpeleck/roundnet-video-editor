import React from 'react';
import { afterEach, expect, test, vi } from 'vitest';
import { act, cleanup, fireEvent, render, screen } from '@testing-library/react';
import { Setup } from './Setup';
import { pickFile } from './api';

vi.mock('./api', async importOriginal => ({ ...await importOriginal<typeof import('./api')>(), pickFile: vi.fn() }));
afterEach(() => { cleanup(); vi.clearAllMocks(); vi.useRealTimers(); });

test('a recording selection can take longer than five minutes and still finish', async () => {
  vi.useFakeTimers();
  let finish!: (path: string) => void;
  vi.mocked(pickFile).mockReturnValue(new Promise(resolve => { finish = resolve; }));
  const onError = vi.fn();
  render(<Setup project={null} command={vi.fn()} update={vi.fn()} onError={onError} onBack={vi.fn()}/>);
  fireEvent.click(screen.getByRole('button', { name: /Select a recording/ }));
  await act(async () => { vi.advanceTimersByTime(301_000); });
  expect(screen.getByRole('status').textContent).toContain('Large files can take a while');
  expect((screen.getByRole('button', { name: /Waiting for your recording/ }) as HTMLButtonElement).disabled).toBe(true);
  expect(onError).not.toHaveBeenCalled();
  await act(async () => { finish('/recordings/large game.mov'); });
  expect((screen.getByLabelText('Recording path') as HTMLInputElement).value).toBe('/recordings/large game.mov');
  expect(screen.queryByRole('status')).toBeNull();
  expect((screen.getByRole('button', { name: /Continue/ }) as HTMLButtonElement).disabled).toBe(false);
});

test('cancelling a recording selection clears its waiting state', async () => {
  vi.mocked(pickFile).mockResolvedValue('');
  const onError = vi.fn();
  render(<Setup project={null} command={vi.fn()} update={vi.fn()} onError={onError} onBack={vi.fn()}/>);
  await act(async () => { fireEvent.click(screen.getByRole('button', { name: /Select a recording/ })); });
  expect(screen.queryByRole('status')).toBeNull();
  expect((screen.getByRole('button', { name: /Select a recording/ }) as HTMLButtonElement).disabled).toBe(false);
  expect(onError).not.toHaveBeenCalled();
});
