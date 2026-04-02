"use client";

import React, { useState } from 'react';
import { useRouter } from 'next/navigation';
import { Upload } from 'lucide-react';
import Layout from '../../components/Layout';
import Card from '../../components/ui/Card';
import Toggle from '../../components/ui/Toggle';
import Button from '../../components/ui/Button';

const Detector = () => {
  const router = useRouter();
  const [aiDetection, setAiDetection] = useState(true);
  const [misinformation, setMisinformation] = useState(true);
  const [text, setText] = useState('');

  return (
    <Layout>
      <div className="max-w-4xl mx-auto">
        <h1 className="text-3xl font-bold text-gray-800 mb-8">AI Content & Misinformation Detector</h1>

        <div className="flex flex-col md:flex-row gap-8">
          {/* Main Panel */}
          <Card className="flex-1 flex flex-col min-h-[500px]">
            <div className="flex items-center gap-6 mb-6">
              <Toggle
                label="AI Generated Detection (ON)"
                enabled={aiDetection}
                onChange={setAiDetection}
              />
              <Toggle
                label="Misinformation Signals (ON)"
                enabled={misinformation}
                onChange={setMisinformation}
              />
            </div>

            <textarea
              className="flex-1 w-full p-6 text-gray-700 placeholder:text-gray-400 bg-gray-50/50 rounded-xl border border-gray-100 focus:outline-none resize-none text-lg leading-relaxed"
              placeholder="Enter or paste the text you want to verify here..."
              value={text}
              onChange={(e) => setText(e.target.value)}
            />

            <div className="mt-6 flex items-center justify-between">
              <Button variant="outline" className="flex items-center gap-2">
                <Upload size={18} />
                Upload File
              </Button>
            </div>
          </Card>
        </div>

        <div className="mt-10 flex justify-center">
          <Button
            className="px-12 py-4 text-lg font-bold shadow-lg shadow-primary-green/20"
            onClick={() => router.push('/result')}
          >
            Analyze Now
          </Button>
        </div>
      </div>
    </Layout>
  );
};

export default Detector;
