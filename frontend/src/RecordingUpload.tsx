import { useEffect, useRef, useState } from 'react';
import { uploadRecording } from './api';
import { Icon } from './icons';

export function RecordingUpload({ onUploaded, onBusy, onError }: { onUploaded: (path: string) => void; onBusy: (busy: boolean) => void; onError: (e: unknown) => void }) {
  const [progress, setProgress] = useState<number | null>(null);
  const [name, setName] = useState('');
  const controller = useRef<AbortController | null>(null);
  useEffect(() => () => controller.current?.abort(), []);
  const select = async (file: File) => {
    const transfer = new AbortController(); controller.current = transfer;
    onBusy(true); setProgress(0);
    try {
      const result = await uploadRecording(file, setProgress, transfer.signal);
      if (!transfer.signal.aborted) { setName(file.name); onUploaded(result.path); }
    } catch (e) { if (!transfer.signal.aborted) onError(e); }
    finally { if (controller.current === transfer) { controller.current = null; setProgress(null); onBusy(false); } }
  };
  return progress === null ? <><label className="file-drop phone-file-drop"><span className="file-icon"><Icon name="video" size={32}/></span><strong>{name || 'Select a recording'}</strong><span>{name ? 'Ready on your Mac · tap to choose another' : 'Choose a video from Photos or Files'}</span><span className="button">Choose video</span><input type="file" accept="video/*,.mov,.mp4,.m4v,.mkv,.avi,.mts,.m2ts,.webm" aria-label="Choose video from your phone" onChange={e => { const file = e.target.files?.[0]; e.target.value = ''; if (file) void select(file); }}/></label><p className="helper">Your recording transfers directly to your Mac. Keep Safari open during the transfer.</p></> : <div className="phone-transfer" role="status"><strong>{progress === 100 ? 'Checking your recording…' : `Transferring to your Mac… ${progress}%`}</strong><progress value={progress} max={100}/><p>Keep Safari open and your Mac awake.</p><button className="button" onClick={() => controller.current?.abort()}>Cancel transfer</button></div>;
}
