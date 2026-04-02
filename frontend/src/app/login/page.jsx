"use client";

import React from 'react';
import { useRouter } from 'next/navigation';
import Card from '../../components/ui/Card';
import Input from '../../components/ui/Input';
import Button from '../../components/ui/Button';

const Login = () => {
  const router = useRouter();

  return (
    <div className="min-h-screen bg-bg-grey flex items-center justify-center p-4">
      <Card className="w-full max-w-[350px] p-[30px] flex flex-col items-center">
        <div className="text-center mb-8">
          <h1 className="text-2xl font-bold text-gray-800">Truthness</h1>
          <p className="text-sm text-gray-500 font-medium">An AI detector System</p>
        </div>

        <div className="w-full space-y-4 mb-8">
          <Input label="Username" placeholder="Enter your username" />
          <Input label="Password" type="password" placeholder="Enter your password" />
        </div>

        <div className="w-full flex gap-3 mb-6">
          <Button className="flex-1" onClick={() => router.push('/dashboard')}>Login</Button>
          <Button variant="danger" className="flex-1" onClick={() => router.push('/register')}>Register</Button>
        </div>

        <button className="text-sm text-gray-400 hover:text-gray-600 transition-colors">
          Select Language
        </button>
      </Card>
    </div>
  );
};

export default Login;
