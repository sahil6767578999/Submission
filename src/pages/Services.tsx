import React from "react";
import Header from "@/components/Header";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Star, Moon, Sun } from "lucide-react";

const Services = () => {
  return (
    <div className="min-h-screen">
      <Header />
      <main className="container mx-auto px-4 py-8">
        <h1 className="text-4xl font-display text-white mb-6">Our Services</h1>
        <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
          <ServiceCard
            icon={<Star className="text-primary w-6 h-6" />}
            title="Astrological Readings"
            description="Personalized horoscope and celestial guidance"
          />
          <ServiceCard
            icon={<Moon className="text-primary w-6 h-6" />}
            title="Chakra Healing"
            description="Balance and align your energy centers"
          />
          <ServiceCard
            icon={<Sun className="text-primary w-6 h-6" />}
            title="Spiritual Counseling"
            description="One-on-one guidance for your spiritual journey"
          />
        </div>
      </main>
    </div>
  );
};

const ServiceCard = ({ icon, title, description }: { icon: React.ReactNode; title: string; description: string }) => (
  <Card className="bg-white/80 backdrop-blur-sm">
    <CardHeader>
      <CardTitle className="flex items-center gap-2">
        {icon}
        {title}
      </CardTitle>
    </CardHeader>
    <CardContent>
      <p className="text-gray-700">{description}</p>
    </CardContent>
  </Card>
);

export default Services;