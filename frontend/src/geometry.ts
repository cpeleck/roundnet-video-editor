import type { Project } from './types';
export function displayDimensions(project: Project): [number, number] {
  const { width = 16, height = 9, rotation_degrees = 0 } = project.video_metadata;
  return Math.abs(rotation_degrees) % 180 === 90 ? [height, width] : [width, height];
}
