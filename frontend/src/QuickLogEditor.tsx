import { useState } from 'react';
import type { Assignment, Project, QuickLog, Rally } from './types';
import { labelFor } from './App';

export const emptyQuickLog = (): QuickLog => ({ version: 1, possessions: [], serve_faults: [], terminal_defender_id: null, error: null });
export function QuickLogEditor({ project, rally, assignment, kind, busy, onSave, onCancel, onReplay }: {
  project: Project; rally: Rally; assignment: Assignment; kind: string; busy: boolean;
  onSave: (log: QuickLog) => Promise<void>; onCancel: () => void; onReplay: () => void;
}) {
  const players = project.match_settings.players;
  const rallyKind = ['defensive_hold', 'defensive_break', 'error'].includes(kind);
  const [log, setLog] = useState<QuickLog>(() => rally.classification.kind === kind && rally.point_stats.quick_log
    ? structuredClone(rally.point_stats.quick_log) : { ...emptyQuickLog(), possessions: rallyKind ? [{ first_touch_id: assignment.receiver_id, hitter_id: assignment.receiver_id }] : [], error: kind === 'error' ? { player_id: '', kind: 'hit' } : null });
  const [history, setHistory] = useState<QuickLog[]>([]);
  const [saving, setSaving] = useState(false);
  const change = (next: QuickLog) => { setHistory([...history, structuredClone(log)]); setLog(next); };
  const last = log.possessions.at(-1);
  const winningTeam = kind === 'defensive_hold' ? assignment.server_id[0] : assignment.receiver_id[0];
  const finalTeam = log.possessions.length % 2 ? assignment.receiver_id[0] : assignment.server_id[0];
  const ready = !rallyKind || (log.possessions.every(p => p.first_touch_id && p.hitter_id) && (kind === 'error'
    ? !!log.error?.player_id && log.error.player_id[0] === finalTeam && (log.error.kind !== 'receive' || log.possessions.length === 1 && log.error.player_id === assignment.receiver_id)
    : log.possessions.length >= 2 && last?.hitter_id[0] === winningTeam));
  const options = (team: string) => players.filter(p => p.team === team).map(p => <option key={p.player_id} value={p.player_id}>{p.name}</option>);
  return <section className="quick-log" aria-label="Guided possession log"><h2>{labelFor(kind)} · Player details</h2><p className="helper">Follow each possession in order. The first-touch player is the suggested hitter; correct hits on one or two. Ungraded touches stay ungraded.</p><button className="button small" onClick={onReplay} type="button">Replay this clip</button>
    <fieldset disabled={busy || saving}>
      {rallyKind && <><ol className="possession-list">{log.possessions.map((p, i) => {
        const team = i % 2 ? assignment.server_id[0] : assignment.receiver_id[0];
        const update = (fields: Partial<typeof p>) => change({ ...log, possessions: log.possessions.map((row, j) => j === i ? { ...row, ...fields } : row) });
        return <li key={i}><strong>{i + 1}. {i ? 'Defensive possession' : 'Serve receive'} · {team === 'A' ? project.match_settings.team_a : project.match_settings.team_b}</strong>
          <label>First touch {i + 1}<select aria-label={`First touch ${i + 1}`} value={p.first_touch_id} disabled={i === 0} onChange={e => update({ first_touch_id: e.target.value, hitter_id: e.target.value })}><option value="">Choose player</option>{options(team)}</select></label>
          <label>{i === log.possessions.length - 1 && kind !== 'error' ? 'Who puts it away?' : 'Hitter'}<select aria-label={`Hitter ${i + 1}`} value={p.hitter_id} onChange={e => update({ hitter_id: e.target.value })}><option value="">Choose player</option>{options(team)}</select></label>
          <details><summary>Costly touch corrections</summary>{i === 0 && <label className="checkbox"><input type="checkbox" checked={!!p.tough_receive} onChange={e => update({ tough_receive: e.target.checked })}/>Costly weak receive</label>}<label className="checkbox"><input type="checkbox" checked={!!p.tough_set} onChange={e => update({ tough_set: e.target.checked })}/>Costly weak set by partner</label></details>
        </li>;
      })}</ol><div className="row"><button className="button small" disabled={log.possessions.length >= 200} onClick={() => change({ ...log, terminal_defender_id: null, possessions: [...log.possessions, { first_touch_id: '', hitter_id: '' }] })}>Add possession</button><button className="button small" disabled={log.possessions.length <= 1} onClick={() => change({ ...log, terminal_defender_id: null, possessions: log.possessions.slice(0, -1) })}>Remove last</button></div>
      {kind === 'error' ? <div className="terminal-entry"><label>Who made the error?<select aria-label="Error player" value={log.error?.player_id ?? ''} onChange={e => {
        const player_id = e.target.value;
        change({ ...log, error: { player_id, kind: log.error?.kind ?? 'hit' }, possessions: log.possessions.map((p, i) => i === log.possessions.length - 1 && log.error?.kind === 'hit' ? { ...p, hitter_id: player_id } : p) });
      }}><option value="">Choose player</option>{options(finalTeam)}</select></label><label>Error touch<select aria-label="Error touch" value={log.error?.kind ?? 'hit'} onChange={e => change({ ...log, error: { player_id: log.error?.player_id ?? '', kind: e.target.value }, possessions: log.possessions.map((p, i) => i === log.possessions.length - 1 && e.target.value === 'hit' && log.error?.player_id ? { ...p, hitter_id: log.error.player_id } : p) })}><option value="receive" disabled={log.possessions.length !== 1}>Receive</option><option value="set">Set</option><option value="hit">Hit</option></select></label></div>
        : <label className="terminal-entry">After the winning hit, did an opponent touch it?<select aria-label="Final defender" value={log.terminal_defender_id ?? ''} onChange={e => change({ ...log, terminal_defender_id: e.target.value || null })}><option value="">No touch</option>{options(finalTeam === 'A' ? 'B' : 'A')}</select></label>}
      {!ready && <p className="helper">{kind === 'error' ? 'Choose each first-touch player and the final error details.' : `Add each possession, then select a finisher on ${winningTeam === 'A' ? project.match_settings.team_a : project.match_settings.team_b}.`}</p>}</>}
      <details><summary>Serve corrections</summary><p className="helper">Default: one successful serve, no costly weak touches. Double Fault already records two faults.</p><div className="row">{['fault', 'rim', 'let'].map(f => <button className="button small" key={f} disabled={kind === 'double_fault' || log.serve_faults.length >= 50 || f !== 'let' && log.serve_faults.some(v => v !== 'let')} onClick={() => change({ ...log, serve_faults: [...log.serve_faults, f] })}>Add {f}</button>)}</div>{!!log.serve_faults.length && <p>{log.serve_faults.map(labelFor).join(' → ')} → live serve <button className="text-button" onClick={() => change({ ...log, serve_faults: log.serve_faults.slice(0, -1) })}>Remove last serve correction</button></p>}</details>
      <p className="helper">{rallyKind ? 'Ordinary three-touch possessions suggest a partner set. You can correct the full touch log later.' : 'This outcome assumes no rally. Use detailed touches to correct player roles.'}</p>
      <div className="row"><button className="button small" disabled={!history.length} onClick={() => { setLog(history.at(-1)!); setHistory(history.slice(0, -1)); }}>Undo entry</button><button className="button" onClick={onCancel}>Cancel</button><button className="button primary" disabled={!ready} onClick={async () => { setSaving(true); try { await onSave(log); } finally { setSaving(false); } }}>Save point & continue</button></div>
    </fieldset>
  </section>;
}
