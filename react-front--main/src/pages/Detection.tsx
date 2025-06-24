import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { Upload, Loader2 } from 'lucide-react';

export default function Detection() {
  const [isProcessing, setIsProcessing] = useState(false);
  const navigate = useNavigate();

  const handleUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    if (!e.target.files?.length) return;
    
    setIsProcessing(true);
    // Simulate processing
    setTimeout(() => {
      setIsProcessing(false);
      navigate('/results');
    }, 2000);
  };

  return (
    <div className="min-h-screen bg-gray-50 p-4">
      <div className="max-w-3xl mx-auto">
        <div className="bg-white rounded-xl shadow-md p-8">
          <h2 className="text-2xl font-bold text-center mb-8">Upload Media for Analysis</h2>

          <div className="border-2 border-dashed border-gray-300 rounded-lg p-12">
            <div className="flex flex-col items-center">
              {isProcessing ? (
                <>
                  <Loader2 size={48} className="text-blue-600 animate-spin mb-4" />
                  <p className="text-xl font-medium">Processing your media...</p>
                  <p className="text-gray-500 mt-2">This may take a few moments</p>
                </>
              ) : (
                <>
                  <Upload size={48} className="text-blue-600 mb-4" />
                  <p className="text-xl font-medium">Drag and drop your file here</p>
                  <p className="text-gray-500 mt-2">or</p>
                  <label className="mt-4">
                    <input
                      type="file"
                      className="hidden"
                      accept="image/*,video/*"
                      onChange={handleUpload}
                    />
                    <span className="px-4 py-2 bg-blue-600 text-white rounded-md cursor-pointer hover:bg-blue-700">
                      Browse Files
                    </span>
                  </label>
                  <p className="text-sm text-gray-500 mt-4">
                    Supported formats: JPG, PNG, MP4, MOV
                  </p>
                </>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}