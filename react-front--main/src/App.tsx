import React from "react";
import { BrowserRouter as Router, Routes, Route, Navigate } from "react-router-dom";
import SplashScreen from "./pages/SplashScreen";
import Auth from "./pages/Auth";
import Home from "./pages/Home";
import Detection from "./pages/Detection";
import Results from "./pages/Results";
import Upload from "./pages/Upload";
import LiveDetection from "./pages/LiveDetection";  // New component
import 'bootstrap/dist/css/bootstrap.min.css';

const App: React.FC = () => {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<SplashScreen />} />
        <Route path="/auth" element={<Auth />} />
        <Route path="/home" element={<Home />} />
        <Route path="/detection" element={<Detection />} />
        <Route path="/results" element={<Results />} />
        <Route path="/upload" element={<Upload />} />
        <Route path="/live-detection" element={<LiveDetection />} /> {/* Live Detection */}
        <Route path="*" element={<Navigate to="/upload" replace />} />
      </Routes>
    </Router>
  );
};

export default App;
