import { Suspense } from "react";
import { Routes, Route, Navigate } from "react-router-dom";
import NavbarProvider from "./components/layout/NavbarProvider";

// Import dashboard components
import Layout from "./components/layout/Layout";
import OverviewDashboard from "./components/dashboard/OverviewDashboard";
import AnalyticsDashboard from "./components/dashboard/AnalyticsDashboard";
import ClassificationDashboard from "./components/dashboard/ClassificationDashboard";
import PredictionDashboard from "./components/dashboard/PredictionDashboard";
import ViolationsDashboard from "./components/dashboard/ViolationsDashboard";

function App() {
  return (
    <NavbarProvider>
      <Suspense
        fallback={
          <div className="flex items-center justify-center h-screen">
            Loading...
          </div>
        }
      >
        <Routes>
          {/* Dashboard routes - all using consistent Layout */}
          <Route path="/" element={
            <Layout title="🕺BitDance">
              <OverviewDashboard />
            </Layout>
          } />

          <Route path="/analytics" element={         
            <Layout title="Analytics">
              <AnalyticsDashboard />
            </Layout>
          } />

          <Route path="/classification" element={
            <Layout title="Classification">
              <ClassificationDashboard />
            </Layout>
          } />

          <Route path="/prediction" element={
            <Layout title="Prediction">
              <PredictionDashboard />
            </Layout>
          } />

          <Route path="/violations" element={
            <Layout title="Violations">
              <ViolationsDashboard />
            </Layout>
          } />

          {/* Redirect any unknown routes to home */}
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </Suspense>
    </NavbarProvider>
  );
}

export default App;
