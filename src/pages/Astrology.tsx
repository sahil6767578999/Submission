import React from "react";
import Header from "@/components/Header";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

const Astrology = () => {
  return (
    <div className="min-h-screen">
      <Header />
      <main className="container mx-auto px-4 py-8">
        <h1 className="text-4xl font-display text-white mb-6">Astrology 101</h1>
        <div className="bg-white/80 backdrop-blur-sm rounded-2xl p-8">
          <div className="space-y-6">
            <section>
              <h2 className="text-2xl font-display mb-4">Understanding the Basics</h2>
              <p className="text-gray-700">
                Astrology is the study of the movements and relative positions of celestial bodies, such as the sun, moon, planets, and stars, and their influence on human affairs and terrestrial events.
              </p>
            </section>
            <section>
              <h2 className="text-2xl font-display mb-4">The Zodiac Signs</h2>
              <p className="text-gray-700">
                The zodiac is divided into twelve signs, each representing different personality traits, strengths, and challenges. Understanding your sun sign is the first step in your astrological journey.
              </p>
            </section>
          </div>
        </div>
      </main>
    </div>
  );
};

export default Astrology;