import React from 'react';
import Navbar from './Navbar';

const Layout = ({ children }) => {
  return (
    <div className="min-h-screen bg-bg-grey flex flex-col">
      <Navbar />
      <main className="flex-1 p-10 max-w-[1400px] mx-auto w-full">
        {children}
      </main>
    </div>
  );
};

export default Layout;
