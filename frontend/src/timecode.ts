/** Display source times with minute rollover handled before formatting seconds. */
export function formatTimecode(time: number, decimals = 1) {
  const scale = 10 ** decimals;
  const ticks = Math.round(Math.max(0, time) * scale);
  const minutes = Math.floor(ticks / (60 * scale));
  const seconds = (ticks % (60 * scale)) / scale;
  return `${String(minutes).padStart(2, '0')}:${seconds.toFixed(decimals).padStart(decimals ? 3 + decimals : 2, '0')}`;
}

export const formatTrimTimecode = (time: number) => formatTimecode(time, 6).replace(/\.?0+$/, '');

/** Minutes may exceed 59; seconds must be 00–59 with optional fractions. */
export function parseTimecode(value: string): number | null {
  const match = /^(\d+):([0-5]\d(?:\.\d+)?)$/.exec(value.trim());
  if (!match) return null;
  const seconds = Number(match[1]) * 60 + Number(match[2]);
  return Number.isFinite(seconds) ? seconds : null;
}
