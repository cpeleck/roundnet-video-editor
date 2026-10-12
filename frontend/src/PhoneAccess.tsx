import { useEffect, useRef, useState, type ReactNode } from 'react';
import { isPhoneConnection, request } from './api';

export function PhoneAccess({ children }: { children: ReactNode }) {
  const initialToken = useRef(new URLSearchParams(window.location.hash.slice(1)).get('pair'));
  const shouldPair = isPhoneConnection() || !!initialToken.current;
  const [ready, setReady] = useState(!shouldPair);
  const [checking, setChecking] = useState(true);
  const [error, setError] = useState('');
  const [link, setLink] = useState('');
  const pair = async (token: string) => {
    await request('/session', { token });
    window.history.replaceState(null, '', window.location.pathname + window.location.search);
  };
  useEffect(() => {
    if (!shouldPair) return;
    let active = true;
    void (async () => {
      try {
        if (initialToken.current) await pair(initialToken.current);
        const session = await request<{ paired: boolean }>('/session');
        if (active) setReady(session.paired);
      } catch (e) { if (active) setError(e instanceof Error ? e.message : String(e)); }
      finally { if (active) setChecking(false); }
    })();
    return () => { active = false; };
  }, [shouldPair]);
  if (ready) return children;
  return <main className="pair-page"><div className="eyebrow">ROUNDNET ON YOUR PHONE</div><h1>{checking ? 'Connecting…' : 'Connect to your Mac.'}</h1><p>Open the private pairing link shown on your Mac. Keep both devices on the same Wi-Fi.</p>{checking ? <span className="spinner"/> : <form onSubmit={e => { e.preventDefault(); setChecking(true); setError(''); let token = link.trim(); try { token = new URLSearchParams(new URL(token).hash.slice(1)).get('pair') ?? ''; } catch { /* Accept a copied code too. */ } void pair(token).then(() => setReady(true)).catch(e => setError(e.message)).finally(() => setChecking(false)); }}><label>Pairing link or code<input value={link} onChange={e => setLink(e.target.value)} autoCapitalize="none" autoCorrect="off" spellCheck={false}/></label><button className="button primary wide" disabled={!link.trim()}>Connect</button></form>}{error && <p role="alert">{error}</p>}</main>;
}
