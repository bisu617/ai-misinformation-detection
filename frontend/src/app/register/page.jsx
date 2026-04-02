"use client";

import React, { useState } from 'react';
import { useRouter } from 'next/navigation';
import Card from '../../components/ui/Card';
import Input from '../../components/ui/Input';
import Button from '../../components/ui/Button';

const Register = () => {
  const router = useRouter();
  const [formData, setFormData] = useState({
    username: '',
    password: '',
    fullName: '',
    email: '',
    phone: ''
  });

  const handleChange = (e, field) => {
    setFormData({ ...formData, [field]: e.target.value });
  };

  const handleCreateAccount = () => {
    const { username, password, fullName, email, phone } = formData;
    if (!username || !password || !fullName || !email || !phone) {
      alert("Please fill all basic information fields");
      return;
    }
    alert("Account created successfully!");
    router.push('/login');
  };

  return (
    <div className="min-h-screen bg-bg-grey flex items-center justify-center p-6">
      <Card className="w-full max-w-[600px] p-8">
        <div className="space-y-8">
          {/* Section 1: Basic Information */}
          <div className="border-2 border-blue-100 rounded-xl p-6">
            <h2 className="text-lg font-bold text-gray-800 mb-4">Basic Information</h2>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <Input label="Username" placeholder="Choose a username" value={formData.username} onChange={(e) => handleChange(e, 'username')} />
              <Input label="Password" type="password" placeholder="Choose a password" value={formData.password} onChange={(e) => handleChange(e, 'password')} />
              <div className="md:col-span-2">
                <Input label="Full Name" placeholder="Your full name" value={formData.fullName} onChange={(e) => handleChange(e, 'fullName')} />
              </div>
              <Input label="Email" type="email" placeholder="your@email.com" value={formData.email} onChange={(e) => handleChange(e, 'email')} />
              <Input label="Phone Number" placeholder="+1 (555) 000-0000" value={formData.phone} onChange={(e) => handleChange(e, 'phone')} />
            </div>
          </div>

          {/* Section 2: Account Type */}
          <div className="border-2 border-orange-100 rounded-xl p-6">
            <h2 className="text-lg font-bold text-gray-800 mb-4">Account Type</h2>
            <div className="w-full">
              <label className="block text-sm font-medium text-gray-700 mb-1">Select Role</label>
              <select className="w-full px-4 py-2 rounded-lg border border-border-grey focus:ring-2 focus:ring-primary-green focus:border-transparent outline-none transition-all">
                <option value="student">Student</option>
                <option value="teacher">Teacher</option>
              </select>
            </div>
          </div>

          <div className="flex gap-4">
            <Button className="flex-1" onClick={handleCreateAccount}>Create Account</Button>
            <Button variant="secondary" className="flex-1" onClick={() => router.push('/login')}>Back to Login</Button>
          </div>
        </div>
      </Card>
    </div>
  );
};

export default Register;
