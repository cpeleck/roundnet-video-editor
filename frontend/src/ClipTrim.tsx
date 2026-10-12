import { useId, useState } from 'react';
import { formatTrimTimecode, parseTimecode } from './timecode';

export function ClipTrim({ startTime, endTime, duration, busy, position, saveLabel = 'Save trim', onSave }: {
  startTime: number; endTime: number; duration: number; busy: boolean; position?: number;
  saveLabel?: string; onSave: (start: number, end: number) => Promise<unknown>;
}) {
  const [startText, setStartText] = useState(() => formatTrimTimecode(startTime));
  const [endText, setEndText] = useState(() => formatTrimTimecode(endTime));
  const [saving, setSaving] = useState(false);
  const helpId = useId();
  // Preserve original precision for boundaries the user has not changed.
  const start = startText === formatTrimTimecode(startTime) ? startTime : parseTimecode(startText);
  const end = endText === formatTrimTimecode(endTime) ? endTime : parseTimecode(endText);
  const error = start === null || end === null ? 'Enter times as MM:SS, for example 01:23 or 01:23.45.'
    : end <= start ? 'The clip end must be after its start.'
    : end > duration ? 'The clip end cannot be beyond the recording.' : '';
  const disabled = busy || saving;
  return <form onSubmit={async e => {
    e.preventDefault();
    if (disabled || error || start === null || end === null) return;
    setSaving(true);
    try { await onSave(start, end); } finally { setSaving(false); }
  }}>
    <div className="row">
      <label>In (MM:SS)<input type="text" value={startText} disabled={disabled} spellCheck={false} placeholder="00:00" aria-label="In (MM:SS)" aria-invalid={start === null} aria-describedby={helpId} onChange={e => setStartText(e.target.value)}/>
        {position !== undefined && <button type="button" className="text-button" disabled={disabled} onClick={() => setStartText(formatTrimTimecode(position))}>Use playhead</button>}
      </label>
      <label>Out (MM:SS)<input type="text" value={endText} disabled={disabled} spellCheck={false} placeholder="00:00" aria-label="Out (MM:SS)" aria-invalid={end === null} aria-describedby={helpId} onChange={e => setEndText(e.target.value)}/>
        {position !== undefined && <button type="button" className="text-button" disabled={disabled} onClick={() => setEndText(formatTrimTimecode(position))}>Use playhead</button>}
      </label>
    </div>
    <p id={helpId} className={`helper ${error ? 'warning-text' : ''}`} role={error ? 'alert' : undefined}>{error || 'Use MM:SS. Add decimal seconds for a precise cut, such as 01:23.45.'}</p>
    <button className="button small wide" type="submit" disabled={disabled || !!error}>{saveLabel}</button>
  </form>;
}
