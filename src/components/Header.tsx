import React, { useState, useEffect } from "react";
import { Moon, Sun } from "lucide-react";
import { Button } from "@/components/ui/button";

const quotes = [
  "As above, so below.",
  "Follow your inner moonlight.",
  "The universe works in mysterious ways.",
  "Trust the timing of your life.",
  "Let your spirit shine bright.",
];

const Header = () => {
  const [currentQuote, setCurrentQuote] = useState(quotes[0]);

  useEffect(() => {
    const interval = setInterval(() => {
      const randomIndex = Math.floor(Math.random() * quotes.length);
      setCurrentQuote(quotes[randomIndex]);
    }, 5000);

    return () => clearInterval(interval);
  }, []);

  return (
    <header className="bg-white/80 backdrop-blur-sm sticky top-0 z-50 border-b border-gray-200">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          {/* Logo */}
          <div className="flex items-center">
            <div className="flex-shrink-0">
              <Moon className="h-8 w-8 text-primary" />
            </div>
            <h1 className="ml-2 text-xl font-display font-bold gradient-text">
              SoulBuddy
            </h1>
          </div>

          {/* Navigation */}
          <nav className="hidden md:flex space-x-8">
            <a href="/" className="text-gray-700 hover:text-primary transition-colors">
              Home
            </a>
            <a href="/about" className="text-gray-700 hover:text-primary transition-colors">
              About Us
            </a>
            <a href="/services" className="text-gray-700 hover:text-primary transition-colors">
              Services
            </a>
            <a href="/astrology" className="text-gray-700 hover:text-primary transition-colors">
              Astrology 101
            </a>
            <a href="/contact" className="text-gray-700 hover:text-primary transition-colors">
              Contact Us
            </a>
          </nav>

          {/* Quote */}
          <div className="hidden lg:block">
            <p className="text-sm text-gray-600 italic animate-fade-in">
              {currentQuote}
            </p>
          </div>

          {/* Theme Toggle (static for now) */}
          <Button variant="ghost" size="icon" className="ml-4">
            <Sun className="h-5 w-5" />
          </Button>
        </div>
      </div>
    </header>
  );
};

export default Header;