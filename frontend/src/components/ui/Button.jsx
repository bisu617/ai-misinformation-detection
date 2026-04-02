import React from 'react';

const Button = ({ children, variant = 'primary', className = '', ...props }) => {
  const variants = {
    primary: 'bg-primary-green text-white hover:bg-opacity-90',
    secondary: 'bg-gray-200 text-gray-700 hover:bg-gray-300',
    danger: 'bg-accent-orange text-white hover:bg-opacity-90',
    outline: 'border border-border-grey bg-transparent hover:bg-bg-grey text-gray-600',
  };

  return (
    <button
      className={`px-6 py-2 rounded-lg font-medium transition-colors ${variants[variant]} ${className}`}
      {...props}
    >
      {children}
    </button>
  );
};

export default Button;
