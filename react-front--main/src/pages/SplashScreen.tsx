import React, { useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { Shield } from 'lucide-react';

export default function SplashScreen() {
  const navigate = useNavigate();

  useEffect(() => {
    const timer = setTimeout(() => {
      navigate('/auth');
    }, 3000);
    return () => clearTimeout(timer);
  }, [navigate]);

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-900 to-purple-900 flex flex-col items-center justify-center">
      <div className="animate-bounce mb-8">
        <Shield size={80} className="text-white" />
      </div>
      <h1 className="text-4xl font-bold text-white mb-4">DeepGuard</h1>
      <p className="text-blue-200 text-lg">Advanced Deep Fake Detection</p>
    </div>
  );
}