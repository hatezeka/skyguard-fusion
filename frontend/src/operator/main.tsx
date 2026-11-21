import React from 'react'
import {createRoot} from 'react-dom/client'
import App from './App'
import './operator.css'
createRoot(document.getElementById('operator-root')!).render(<React.StrictMode><App/></React.StrictMode>)
