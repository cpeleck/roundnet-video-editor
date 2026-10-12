import React from 'react';
import { afterEach, expect, test, vi } from 'vitest';
import { act, cleanup, fireEvent, render, screen } from '@testing-library/react';
import { RecordingUpload } from './RecordingUpload';
import { uploadRecording } from './api';

vi.mock('./api', async importOriginal => ({ ...await importOriginal<typeof import('./api')>(), uploadRecording: vi.fn() }));
afterEach(() => { cleanup(); vi.clearAllMocks(); });

test('a phone recording reports transfer progress before returning its managed Mac path', async () => {
  let finish!: (value: { path: string }) => void;
  vi.mocked(uploadRecording).mockImplementation((_file, progress) => { progress(45); return new Promise(resolve => { finish = resolve; }); });
  const onUploaded = vi.fn(); const onBusy = vi.fn(); const onError = vi.fn();
  render(<RecordingUpload onUploaded={onUploaded} onBusy={onBusy} onError={onError}/>);
  fireEvent.change(screen.getByLabelText('Choose video from your phone'), { target: { files: [new File(['video'], 'game.mov', { type: 'video/quicktime' })] } });
  expect(screen.getByRole('status').textContent).toContain('45%');
  expect(onUploaded).not.toHaveBeenCalled();
  expect(onBusy).toHaveBeenLastCalledWith(true);
  await act(async () => { finish({ path: '/managed/recordings/game.mov' }); });
  expect(onUploaded).toHaveBeenCalledWith('/managed/recordings/game.mov');
  expect(screen.getByText('game.mov')).toBeTruthy();
  expect(onBusy).toHaveBeenLastCalledWith(false);
  expect(onError).not.toHaveBeenCalled();
});

test('cancelling a phone upload aborts the request and allows selecting another recording', async () => {
  let signal!: AbortSignal;
  vi.mocked(uploadRecording).mockImplementation((_file, _progress, value) => {
    signal = value; return new Promise((_resolve, reject) => value.addEventListener('abort', () => reject(new DOMException('Cancelled', 'AbortError'))));
  });
  const onError = vi.fn(); const onUploaded = vi.fn(); const onBusy = vi.fn();
  render(<RecordingUpload onUploaded={onUploaded} onBusy={onBusy} onError={onError}/>);
  fireEvent.change(screen.getByLabelText('Choose video from your phone'), { target: { files: [new File(['video'], 'game.mov')] } });
  await act(async () => { fireEvent.click(screen.getByRole('button', { name: 'Cancel transfer' })); });
  expect(signal.aborted).toBe(true);
  expect(onUploaded).not.toHaveBeenCalled();
  expect(onError).not.toHaveBeenCalled();
  expect(onBusy).toHaveBeenLastCalledWith(false);
  expect(screen.getByLabelText('Choose video from your phone')).toBeTruthy();
});
