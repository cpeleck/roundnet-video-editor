import { expect, test } from 'vitest';
import { formatTimecode, formatTrimTimecode, parseTimecode } from './timecode';

test.each([
  ['00:00', 0], ['01:23', 83], ['1:23.45', 83.45], [' 00:59.123456 ', 59.123456], ['120:05', 7205],
])('parses %s as source seconds', (value, expected) => {
  expect(parseTimecode(value)).toBe(expected);
});

test.each(['', '83', '01:60', '01:2', '-01:23', '01:23.', '01:23:45', 'one:23'])('rejects invalid trim time %s', value => {
  expect(parseTimecode(value)).toBeNull();
});

test('formats trim values with fractions and correct minute rollover', () => {
  expect(formatTrimTimecode(83)).toBe('01:23');
  expect(formatTrimTimecode(83.45)).toBe('01:23.45');
  expect(formatTrimTimecode(59.9999999)).toBe('01:00');
  expect(formatTimecode(59.96)).toBe('01:00.0');
});
