import { useNavigate } from 'react-router-dom';
import { Upload, FileVideo2, Camera } from 'lucide-react';

export default function Home() {
  const navigate = useNavigate();

  return (
    <div className="min-h-screen bg-gray-50">
      <header className="bg-white shadow">
        <div className="max-w-7xl mx-auto py-6 px-4">
          <h1 className="text-3xl font-bold text-gray-900">DeepGuard Detection</h1>
        </div>
      </header>

      <main className="max-w-7xl mx-auto py-12 px-4">
        <div className="grid md:grid-cols-3 gap-8">
          {/* Upload Image */}
          <div
            onClick={() => navigate('/detection')}
            className="bg-white p-8 rounded-xl shadow-md hover:shadow-lg transition-shadow cursor-pointer"
          >
            <div className="flex flex-col items-center">
              <Upload size={48} className="text-blue-600 mb-4" />
              <h2 className="text-2xl font-semibold mb-2">Upload Image</h2>
              <p className="text-gray-600 text-center">
                Upload an image to detect if it's been manipulated.
              </p>
            </div>
          </div>

          {/* Upload Video */}
          <div
            onClick={() => navigate('/detection')}
            className="bg-white p-8 rounded-xl shadow-md hover:shadow-lg transition-shadow cursor-pointer"
          >
            <div className="flex flex-col items-center">
              <FileVideo2 size={48} className="text-blue-600 mb-4" />
              <h2 className="text-2xl font-semibold mb-2">Upload Video</h2>
              <p className="text-gray-600 text-center">
                Upload a video to analyze for deepfake content.
              </p>
            </div>
          </div>

          {/* Live Detection */}
          <div
            onClick={() => navigate('/live-detection')}
            className="bg-white p-8 rounded-xl shadow-md hover:shadow-lg transition-shadow cursor-pointer"
          >
            <div className="flex flex-col items-center">
              <Camera size={48} className="text-green-600 mb-4" />
              <h2 className="text-2xl font-semibold mb-2">Live Detection</h2>
              <p className="text-gray-600 text-center">
                Use your camera for real-time deepfake detection.
              </p>
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}
