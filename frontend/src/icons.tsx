import type { CSSProperties } from 'react';
const paths: Record<string, string> = {
  arrow: 'M5 12h14m-6-6 6 6-6 6', back: 'm14 6-6 6 6 6', plus: 'M12 5v14M5 12h14',
  play: 'm8 5 11 7-11 7Z', pause: 'M8 5v14M16 5v14', star: 'm12 3 2.8 5.7 6.2.9-4.5 4.4 1.1 6.2-5.6-3-5.6 3 1.1-6.2L3 9.6l6.2-.9Z',
  undo: 'M3 10h7M3 10V3m0 7a8 8 0 1 1 1 9', redo: 'M21 10h-7m7 0V3m0 7a8 8 0 1 0-1 9',
  check: 'm5 12 4 4L19 6', close: 'm6 6 12 12M6 18 18 6', folder: 'M3 7h7l2 3h9v10H3ZM3 7V4h7l2 3h9v3',
  stats: 'M5 20V10m7 10V4m7 16v-7', settings: 'M4 7h16M4 17h16M8 4v6m8 4v6',
  export: 'M12 15V3m-4 4 4-4 4 4M4 13v8h16v-8', video: 'M3 5h13v14H3Zm13 5 5-3v10l-5-3',
  clock: 'M12 7v5l3 2M21 12a9 9 0 1 1-18 0 9 9 0 0 1 18 0', chevron: 'm6 9 6 6 6-6',
  grid: 'M3 3h7v7H3Zm11 0h7v7h-7ZM3 14h7v7H3Zm11 0h7v7h-7Z',
};
export function Icon({ name, size = 18, style }: { name: string; size?: number; style?: CSSProperties }) {
  return <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true" style={style}><path d={paths[name] ?? paths.arrow}/></svg>;
}
