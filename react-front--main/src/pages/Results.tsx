import React from 'react';
import { useNavigate } from 'react-router-dom';
import { CheckCircle2, AlertTriangle } from 'lucide-react';

export default function Results() {
  const navigate = useNavigate();
  const isFake = Math.random() > 0.5; // Simulated result

  return (
    <div className="min-h-screen bg-gray-50 p-4">
      <div className="max-w-3xl mx-auto">
        <div className="bg-white rounded-xl shadow-md p-8">
          <div className="text-center">
            {isFake ? (
              <>
                <AlertTriangle size={64} className="mx-auto text-red-500 mb-4" />
                <h2 className="text-3xl font-bold text-red-500 mb-2">Deep Fake Detected</h2>
                <p className="text-gray-600 mb-8">
                  Our analysis indicates this media has been manipulated
                </p>
              </>
            ) : (
              <>
                <CheckCircle2 size={64} className="mx-auto text-green-500 mb-4" />
                <h2 className="text-3xl font-bold text-green-500 mb-2">Authentic Media</h2>
                <p className="text-gray-600 mb-8">
                  Our analysis indicates this media is authentic
                </p>
              </>
            )}

            <div className="bg-gray-50 p-6 rounded-lg mb-8">
              <h3 className="text-xl font-semibold mb-4">Analysis Details</h3>
              <div className="grid grid-cols-2 gap-4 text-left">
                <div>
                  <p className="text-gray-600">Confidence Score</p>
                  <p className="font-semibold">{isFake ? '92%' : '95%'}</p>
                </div>
                <div>
                  <p className="text-gray-600">Processing Time</p>
                  <p className="font-semibold">2.3 seconds</p>
                </div>
              </div>
            </div>

            <button
              onClick={() => navigate('/home')}
              className="px-6 py-3 bg-blue-600 text-white rounded-md hover:bg-blue-700"
            >
              Analyze Another File
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}