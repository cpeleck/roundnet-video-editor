import React from 'react';
import { afterEach, beforeEach, expect, test, vi } from 'vitest';
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { PhoneAccess } from './PhoneAccess';
import { isPhoneConnection, request } from './api';

vi.mock('./api', async importOriginal => ({ ...await importOriginal<typeof import('./api')>(), request: vi.fn(), isPhoneConnection: vi.fn() }));
beforeEach(() => { vi.mocked(isPhoneConnection).mockReturnValue(true); });
afterEach(() => { cleanup(); vi.clearAllMocks(); window.history.replaceState(null, '', '/'); });

test('pairing link authenticates before opening the editor and removes its code from the URL', async () => {
  window.history.replaceState(null, '', '/#pair=private-code');
  vi.mocked(request).mockImplementation(async (_path, body) => body ? { paired: true } as never : { paired: true } as never);
  render(<PhoneAccess><div>Editor loaded</div></PhoneAccess>);
  await screen.findByText('Editor loaded');
  expect(request).toHaveBeenCalledWith('/session', { token: 'private-code' });
  expect(window.location.hash).toBe('');
});

test('an unpaired device can paste its Mac pairing link without loading project data', async () => {
  vi.mocked(request).mockResolvedValue({ paired: false });
  render(<PhoneAccess><div>Editor loaded</div></PhoneAccess>);
  const input = await screen.findByLabelText('Pairing link or code');
  expect(screen.queryByText('Editor loaded')).toBeNull();
  fireEvent.change(input, { target: { value: 'http://192.168.1.42:5173/#pair=private-code' } });
  vi.mocked(request).mockResolvedValue({ paired: true });
  fireEvent.click(screen.getByRole('button', { name: 'Connect' }));
  await waitFor(() => expect(screen.getByText('Editor loaded')).toBeTruthy());
  expect(request).toHaveBeenCalledWith('/session', { token: 'private-code' });
});

test('ordinary Mac-only startup opens immediately without a pairing request', () => {
  vi.mocked(isPhoneConnection).mockReturnValue(false);
  render(<PhoneAccess><div>Editor loaded</div></PhoneAccess>);
  expect(screen.getByText('Editor loaded')).toBeTruthy();
  expect(request).not.toHaveBeenCalled();
});
