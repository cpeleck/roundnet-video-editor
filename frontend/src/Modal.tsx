import { useEffect, useRef, type ReactNode } from 'react';
import { createPortal } from 'react-dom';

const modalStack: HTMLElement[] = [];
const originalInert = new Map<HTMLElement, boolean>();
let originalOverflow = '';

function updateBackground() {
  const top = modalStack[modalStack.length - 1];
  if (!top) {
    for (const [element, inert] of originalInert) element.inert = inert;
    originalInert.clear();
    document.body.style.overflow = originalOverflow;
    return;
  }
  for (const child of Array.from(document.body.children)) {
    if (!(child instanceof HTMLElement)) continue;
    if (!originalInert.has(child)) originalInert.set(child, child.inert);
    child.inert = child !== top;
  }
}

export function Modal({ children, labelledBy, className = '', onClose }: { children: ReactNode; labelledBy: string; className?: string; onClose: () => void }) {
  const backdrop = useRef<HTMLDivElement>(null);
  const dialog = useRef<HTMLElement>(null);
  const previousFocus = useRef(document.activeElement instanceof HTMLElement ? document.activeElement : null);
  const close = useRef(onClose);
  close.current = onClose;
  useEffect(() => {
    const overlay = backdrop.current!;
    const panel = dialog.current!;
    if (!modalStack.length) originalOverflow = document.body.style.overflow;
    modalStack.push(overlay);
    document.body.style.overflow = 'hidden';
    updateBackground();
    const focusable = () => Array.from(panel.querySelectorAll<HTMLElement>('button, a[href], input, select, textarea, summary, [tabindex]:not([tabindex="-1"])'))
      .filter(element => !element.matches(':disabled') && !element.closest('[inert]') && element.getClientRects().length > 0);
    const focusFirst = () => (focusable()[0] ?? panel).focus({ preventScroll: true });
    if (!panel.contains(document.activeElement)) focusFirst();
    const keydown = (event: KeyboardEvent) => {
      if (modalStack[modalStack.length - 1] !== overlay) return;
      if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === 'z') {
        // Text inputs retain browser undo; application shortcuts cannot edit
        // a point behind an open touch draft or a source-recovery dialog.
        event.stopPropagation();
      }
      if (event.key === 'Escape') {
        event.preventDefault(); event.stopPropagation(); close.current();
      } else if (event.key === 'Tab') {
        const elements = focusable();
        const first = elements[0]; const last = elements[elements.length - 1];
        if (!first) { event.preventDefault(); panel.focus(); }
        else if (event.shiftKey && (document.activeElement === first || !panel.contains(document.activeElement))) {
          event.preventDefault(); last.focus();
        } else if (!event.shiftKey && (document.activeElement === last || !panel.contains(document.activeElement))) {
          event.preventDefault(); first.focus();
        }
      }
    };
    const focusin = (event: FocusEvent) => {
      if (modalStack[modalStack.length - 1] === overlay && !panel.contains(event.target as Node)) focusFirst();
    };
    document.addEventListener('keydown', keydown, true);
    document.addEventListener('focusin', focusin, true);
    return () => {
      document.removeEventListener('keydown', keydown, true);
      document.removeEventListener('focusin', focusin, true);
      modalStack.splice(modalStack.indexOf(overlay), 1);
      updateBackground();
      if (previousFocus.current?.isConnected && !previousFocus.current.closest('[inert]')) previousFocus.current.focus({ preventScroll: true });
    };
  }, []);
  return createPortal(<div className="modal-backdrop" ref={backdrop}><section ref={dialog} className={`modal ${className}`} role="dialog" aria-modal="true" aria-labelledby={labelledBy} tabIndex={-1}>{children}</section></div>, document.body);
}
