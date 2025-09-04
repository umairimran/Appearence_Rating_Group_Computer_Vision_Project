import React, { useState } from 'react';
import './App.css';

// Import pages
import IndividualAnalysis from './pages/IndividualAnalysis';
import GroupAnalysis from './pages/GroupAnalysis';

// Import components
import Tabs from './components/Tabs';

function App() {
  const [activeTab, setActiveTab] = useState('individual');

  const tabs = [
    { id: 'individual', label: 'Individual Analysis (Video)' },
    { id: 'group', label: 'Group Synergy Analysis (Photo)' }
  ];

  return (
    <div className="min-h-screen bg-gray-50 py-8">
      <div className="max-w-4xl mx-auto px-4">
        <h1 className="text-3xl font-bold text-gray-800 text-center mb-6">
          Appearance Rating and Group Synergy AI
        </h1>
        
        <div className="bg-white rounded-2xl shadow-md p-6">
          <Tabs tabs={tabs} activeTab={activeTab} setActiveTab={setActiveTab} />
          
          <div className="mt-6">
            {activeTab === 'individual' && <IndividualAnalysis />}
            {activeTab === 'group' && <GroupAnalysis />}
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;
