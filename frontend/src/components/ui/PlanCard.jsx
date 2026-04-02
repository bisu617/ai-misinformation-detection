import React from 'react';
import { Check } from 'lucide-react';
import Card from './Card';
import Button from './Button';

const PlanCard = ({ title, price, period, features, buttonText, highlighted = false, onSubscribe }) => {
  return (
    <Card className={`flex flex-col h-full flex-1 transition-transform hover:scale-[1.02] ${highlighted ? 'border-primary-green border-2 shadow-lg shadow-primary-green/10' : ''}`}>
      <div className="mb-8">
        <h3 className="text-xl font-bold text-gray-800 mb-2">{title}</h3>
        <div className="flex items-baseline gap-1">
          <span className="text-4xl font-bold text-gray-800">{price}</span>
          <span className="text-gray-500 font-medium">/{period}</span>
        </div>
        {highlighted && (
          <span className="inline-block mt-3 px-3 py-1 bg-primary-green/10 text-primary-green text-xs font-bold rounded-full uppercase tracking-wider">
            Most Popular
          </span>
        )}
      </div>

      <ul className="flex-1 space-y-4 mb-10">
        {features.map((feature, index) => (
          <li key={index} className="flex items-center gap-3 text-sm text-gray-600">
            <div className="bg-primary-green/10 p-1 rounded-full">
              <Check size={14} className="text-primary-green" />
            </div>
            {feature}
          </li>
        ))}
      </ul>

      <Button
        variant={highlighted ? 'primary' : 'outline'}
        className="w-full py-3"
        onClick={onSubscribe}
      >
        {buttonText}
      </Button>
    </Card>
  );
};

export default PlanCard;
