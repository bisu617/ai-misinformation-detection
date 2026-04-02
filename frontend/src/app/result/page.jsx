"use client";

import React, { useState } from 'react';
import { useRouter } from 'next/navigation';
import Layout from '../../components/Layout';
import Card from '../../components/ui/Card';
import Button from '../../components/ui/Button';

const Result = () => {
  const router = useRouter();
  const [text, setText] = useState("");

  const isAnalyzed = text.length > 0;

  const LegendItem = ({ colorClass, label }) => (
    <div className="flex items-center gap-2">
      <div className={`h-1 w-6 rounded-full ${colorClass}`}></div>
      <span className="text-xs text-gray-500">{label}</span>
    </div>
  );

  return (
    <Layout>
      <div className="flex flex-col lg:flex-row gap-8">
        {/* Left Side: Highlighted Text Analysis Panel or Textarea */}
        <div className="flex-[2] flex flex-col gap-6">
          <Card className="flex-1 min-h-[500px] p-0 overflow-hidden relative">
             {!isAnalyzed ? (
                 <textarea
                  className="w-full h-full min-h-[500px] p-6 text-gray-700 placeholder:text-gray-400 bg-gray-50/50 border-none resize-none focus:outline-none text-lg leading-relaxed"
                  placeholder="Paste text here to analyze..."
                  value={text}
                  onChange={(e) => setText(e.target.value)}
                />
             ) : (
                 <div className="p-6">
                    <div className="prose max-w-none text-gray-700 leading-relaxed text-lg mb-6">
                      <p>
                        In the rapidly evolving landscape of <span className="underline decoration-red-400 decoration-2 underline-offset-4">artificial intelligence</span>, the ability to distinguish between <span className="underline decoration-green-400 decoration-2 underline-offset-4">human-written</span> and <span className="underline decoration-yellow-400 decoration-2 underline-offset-4">AI-generated content</span> has become paramount. This platform utilizes <span className="underline decoration-orange-400 decoration-2 underline-offset-4">advanced neural networks</span> to provide a granular analysis of text patterns, ensuring authenticity and trust in digital communication.
                      </p>
                      <p>
                        As we integrate these tools, the risk of <span className="underline decoration-orange-400 decoration-2 underline-offset-4">misinformation</span> grows, making it essential to have a reliable verification system like TruthLens at your disposal.
                      </p>
                    </div>
                 </div>
             )}
          </Card>

          <div className={`flex flex-wrap gap-6 items-center bg-white p-4 rounded-xl shadow-soft border border-gray-100 transition-opacity ${isAnalyzed ? 'opacity-100' : 'opacity-50 grayscale'}`}>
            <LegendItem colorClass="bg-red-400" label="AI Probability" />
            <LegendItem colorClass="bg-green-400" label="Human Probability" />
            <LegendItem colorClass="bg-yellow-400" label="Humanized AI" />
            <LegendItem colorClass="bg-orange-400" label="Misinformation Risk" />
          </div>
        </div>

        {/* Right Side Panel */}
        <div className="flex-1 flex flex-col gap-6">
          <Card className="flex flex-col items-center py-10">
            <h3 className="text-gray-500 font-medium mb-4">Overall Authenticity</h3>
            <div className="relative flex items-center justify-center mb-6">
              <div className={`text-5xl font-bold ${isAnalyzed ? 'text-primary-green' : 'text-gray-300'}`}>
                {isAnalyzed ? '84%' : '0%'}
              </div>
            </div>
            <div className="w-full h-3 bg-gray-100 rounded-full mb-8">
              <div className={`h-full rounded-full transition-all duration-1000 ${isAnalyzed ? 'bg-primary-green' : 'bg-gray-300'}`} style={{ width: isAnalyzed ? '84%' : '0%' }}></div>
            </div>

            <div className="w-full space-y-6">
              <div className="flex justify-between items-center">
                <span className="text-sm font-medium text-gray-500">AI Generated</span>
                <span className={`text-sm font-bold ${isAnalyzed ? 'text-gray-800' : 'text-gray-300'}`}>{isAnalyzed ? '12%' : '0%'}</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-sm font-medium text-gray-500">Human Written</span>
                <span className={`text-sm font-bold ${isAnalyzed ? 'text-gray-800' : 'text-gray-300'}`}>{isAnalyzed ? '78%' : '0%'}</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-sm font-medium text-gray-500">Misinformation Risk</span>
                <span className={`text-sm font-bold uppercase tracking-wider text-xs ${isAnalyzed ? 'text-green-500' : 'text-gray-300'}`}>{isAnalyzed ? 'Low' : 'None'}</span>
              </div>
            </div>
          </Card>

          <Card className={`text-white transition-colors duration-300 ${isAnalyzed ? 'bg-header-grey' : 'bg-gray-300'}`}>
            <h3 className="font-bold mb-2">Detection Tip</h3>
            <p className="text-sm text-white/80 leading-relaxed font-light">
              Sentences with varying lengths and structures are more likely to be human-written. AI models often produce repetitive sentence patterns.
            </p>
          </Card>

          <Button 
            className="w-full py-4 text-lg font-bold" 
            onClick={() => router.push('/report')}
            disabled={!isAnalyzed}
          >
            View Detailed Report
          </Button>
        </div>
      </div>
    </Layout>
  );
};

export default Result;
