import React from "react";
import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom";
import { Toaster } from "sonner";
import { AuthProvider, useAuth } from "@/contexts/AuthContext";
import Layout from "@/components/Layout";
import ProtectedRoute from "@/components/ProtectedRoute";
import Landing from "@/pages/Landing";
import Login from "@/pages/Login";
import Register from "@/pages/Register";
import Screen from "@/pages/Screen";
import Results from "@/pages/Results";
import History from "@/pages/History";
import Learn from "@/pages/Learn";
import ModelsAdmin from "@/pages/ModelsAdmin";

function Root() {
  const { user, loading } = useAuth();
  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center text-[var(--muted)]">
        <div className="font-mono text-sm">Loading…</div>
      </div>
    );
  }
  return user ? <Navigate to="/screen" replace /> : <Landing />;
}

function App() {
  return (
    <AuthProvider>
      <BrowserRouter>
        <Layout>
          <Routes>
            <Route path="/" element={<Root />} />
            <Route path="/login" element={<Login />} />
            <Route path="/register" element={<Register />} />
            <Route path="/learn" element={<Learn />} />
            <Route path="/screen" element={<ProtectedRoute><Screen /></ProtectedRoute>} />
            <Route path="/results/:id" element={<ProtectedRoute><Results /></ProtectedRoute>} />
            <Route path="/history" element={<ProtectedRoute><History /></ProtectedRoute>} />
            <Route path="/admin/models" element={<ProtectedRoute><ModelsAdmin /></ProtectedRoute>} />
            <Route path="*" element={<Navigate to="/" replace />} />
          </Routes>
        </Layout>
        <Toaster position="top-right" richColors closeButton offset={80} />
      </BrowserRouter>
    </AuthProvider>
  );
}

export default App;
