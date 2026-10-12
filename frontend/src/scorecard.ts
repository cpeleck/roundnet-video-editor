import type { Assignment } from './types';

export function scorecardState(assignment: Assignment, completed: boolean) {
  return {
    score: completed ? assignment.score_after ?? assignment.score_before : assignment.score_before,
    server: completed ? assignment.next_server_id ?? assignment.server_id : assignment.server_id,
    receiver: completed ? assignment.next_receiver_id ?? assignment.receiver_id : assignment.receiver_id,
    provisional: completed ? assignment.provisional_after ?? assignment.provisional : assignment.provisional,
  };
}
