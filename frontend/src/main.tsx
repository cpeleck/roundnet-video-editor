import React from 'react';
import { createRoot } from 'react-dom/client';
import App from './App';
import { PhoneAccess } from './PhoneAccess';
import './styles.css';
createRoot(document.getElementById('root')!).render(<React.StrictMode><PhoneAccess><App/></PhoneAccess></React.StrictMode>);
