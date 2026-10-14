import React from 'react'
import { Link } from 'react-router-dom'
import { Card } from '../components/UI'
export default function NotFound(){return <Card><h1>Page not found</h1><p>The requested research workspace page does not exist.</p><Link className="primary-btn" to="/">Back to dashboard</Link></Card>}
