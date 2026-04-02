"use client";

import React from 'react';
import { useRouter } from 'next/navigation';
import { ArrowLeft, Globe, CreditCard, LogOut } from 'lucide-react';
import Button from './ui/Button';

const Navbar = () => {
  const router = useRouter();

  return (
    <nav className="h-[70px] bg-header-grey flex items-center justify-between px-10 text-white sticky top-0 z-50">
      <div className="flex items-center gap-4">
        <button onClick={() => router.back()} className="p-1 hover:bg-white/10 rounded-full transition-colors">
          <ArrowLeft size={20} />
        </button>
        <div>
          <h1 className="font-bold text-lg leading-tight">Welcome Administrator!</h1>
          <p className="text-xs text-white/70 font-light">Dashboard Info</p>
        </div>
      </div>

      <div className="flex items-center gap-3">
        <Button variant="outline" className="text-white border-white/20 hover:bg-white/10 flex items-center gap-2 py-1.5 px-4 h-10">
          <CreditCard size={18} />
          <span>Subscription</span>
        </Button>
        <Button variant="outline" className="text-white border-white/20 hover:bg-white/10 flex items-center gap-2 py-1.5 px-4 h-10">
          <Globe size={18} />
          <span>Language</span>
        </Button>
        <Button variant="danger" className="flex items-center gap-2 py-1.5 px-4 h-10" onClick={() => router.push('/login')}>
          <LogOut size={18} />
          <span>Logout</span>
        </Button>
      </div>
    </nav>
  );
};

export default Navbar;
