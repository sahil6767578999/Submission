import React from "react";
import Header from "@/components/Header";

const About = () => {
  return (
    <div className="min-h-screen">
      <Header />
      <main className="container mx-auto px-4 py-8">
        <h1 className="text-4xl font-display text-white mb-6">About Us</h1>
        <div className="bg-white/80 backdrop-blur-sm rounded-2xl p-8">
          <p className="text-lg text-gray-700 mb-4">
            Welcome to SoulBuddy, your trusted companion on the journey to spiritual enlightenment. We are dedicated to helping you discover your inner truth and connect with the cosmic energies that surround us all.
          </p>
          <p className="text-lg text-gray-700">
            Our team of experienced spiritual guides and astrologists work together to provide you with accurate insights and meaningful guidance for your spiritual journey.
          </p>
        </div>
      </main>
    </div>
  );
};

export default About;