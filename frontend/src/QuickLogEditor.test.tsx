import React from 'react';
import { afterEach, expect, test, vi } from 'vitest';
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { QuickLogEditor } from './QuickLogEditor';
import type { Project, Rally, Assignment } from './types';
afterEach(cleanup);
const assignment = { server_id: 'A1', receiver_id: 'B1' } as Assignment;
const project = { match_settings: { team_a: 'North', team_b: 'South', players: ['A1','A2','B1','B2'].map(player_id => ({ player_id, team: player_id[0], name: player_id })) } } as Project;
const rally = { classification: {}, point_stats: {} } as Rally;
const button = (name: string) => screen.getByRole('button', { name });
for (const kind of ['receive', 'set', 'hit']) test(`Error records its ${kind} type and culprit`, async () => {
  const save = vi.fn().mockResolvedValue(undefined);
  render(<QuickLogEditor project={project} assignment={assignment} rally={rally} kind="error" busy={false} onSave={save} onCancel={() => {}} onReplay={() => {}}/>);
  fireEvent.change(screen.getByLabelText('Error player'), { target: { value: 'B1' } });
  fireEvent.change(screen.getByLabelText('Error touch'), { target: { value: kind } });
  fireEvent.click(button('Save point & continue'));
  await waitFor(() => expect(save).toHaveBeenCalledWith(expect.objectContaining({ error: { player_id: 'B1', kind } })));
});

test('reopening a quick log preserves selections, corrections, and entry undo', () => {
  const old = { ...rally, classification: { kind: 'defensive_hold' }, point_stats: { quick_log: { version: 1, possessions: [{ first_touch_id: 'B1', hitter_id: 'B2' }, { first_touch_id: 'A2', hitter_id: 'A1' }], error: null, terminal_defender_id: 'B1', serve_faults: ['rim'] } } } as Rally;
  render(<QuickLogEditor project={project} assignment={assignment} rally={old} kind="defensive_hold" busy={false} onSave={vi.fn()} onCancel={() => {}} onReplay={() => {}}/>);
  expect((screen.getByLabelText('Hitter 1') as HTMLSelectElement).value).toBe('B2');
  expect((screen.getByLabelText('Hitter 2') as HTMLSelectElement).value).toBe('A1');
  expect((screen.getByLabelText('Final defender') as HTMLSelectElement).value).toBe('B1');
  fireEvent.click(button('Add possession'));
  expect(screen.getByLabelText('First touch 3')).toBeTruthy();
  expect((button('Save point & continue') as HTMLButtonElement).disabled).toBe(true);
  fireEvent.click(button('Undo entry'));
  expect(screen.queryByLabelText('First touch 3')).toBeNull();
  expect((screen.getByLabelText('Final defender') as HTMLSelectElement).value).toBe('B1');
});
