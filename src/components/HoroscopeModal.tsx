import React from "react";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";

interface HoroscopeModalProps {
  isOpen: boolean;
  onClose: () => void;
  sign: string;
}

const HoroscopeModal = ({ isOpen, onClose, sign }: HoroscopeModalProps) => {
  // Extended horoscope data with positive traits and opportunities
  const horoscopeData = {
    daily: {
      Aries: "Today brings exciting opportunities for growth. Trust your instincts and take bold action.",
      Taurus: "Focus on stability and material comfort today. A financial opportunity may present itself.",
      Gemini: "Communication flows easily today. Share your ideas and connect with others.",
      Cancer: "Your intuition is particularly strong today. Listen to your inner voice.",
      Leo: "Leadership opportunities arise today. Stand tall and show your natural confidence.",
      Virgo: "Your analytical skills are heightened. Perfect time for solving complex problems.",
    },
    monthly: {
      Aries: "This month emphasizes personal growth and new beginnings. Your ruling planet Mars brings energy to your endeavors.",
      Taurus: "Venus influences your financial sector this month. Focus on long-term security and investments.",
      Gemini: "Mercury's position brings intellectual stimulation. Great month for learning and communication.",
      Cancer: "The Moon's phases particularly affect you this month. Focus on emotional well-being.",
      Leo: "The Sun empowers your creative pursuits this month. Express yourself boldly.",
      Virgo: "Mercury retrograde brings a time of reflection and revision. Plan carefully.",
    },
    traits: {
      Aries: {
        personal: "Natural leader with boundless energy and enthusiasm",
        relationships: "Passionate and loyal in relationships",
        career: "Excellence in pioneering new projects and taking initiative",
        wellbeing: "Thrives in active, dynamic environments"
      },
      Taurus: {
        personal: "Grounded and reliable with strong values",
        relationships: "Devoted and nurturing partner",
        career: "Exceptional financial acumen and work ethic",
        wellbeing: "Natural connection to physical wellness"
      },
      Gemini: {
        personal: "Quick-witted and adaptable mind",
        relationships: "Engaging and intellectually stimulating companion",
        career: "Versatile skills and excellent communication",
        wellbeing: "Mental agility and social vitality"
      },
      Cancer: {
        personal: "Deep emotional intelligence and intuition",
        relationships: "Nurturing and protective of loved ones",
        career: "Strong memory and project management skills",
        wellbeing: "Natural healing and nurturing abilities"
      },
      Leo: {
        personal: "Charismatic and confident nature",
        relationships: "Generous and warm-hearted friend",
        career: "Natural leadership and creative talents",
        wellbeing: "Radiates positive energy and vitality"
      },
      Virgo: {
        personal: "Detail-oriented and analytical mind",
        relationships: "Thoughtful and supportive partner",
        career: "Excellence in organization and problem-solving",
        wellbeing: "Holistic approach to health and wellness"
      }
    }
  };

  return (
    <Dialog open={isOpen} onOpenChange={onClose}>
      <DialogContent className="bg-white/90 text-gray-800 backdrop-blur-lg border border-primary/20 max-w-2xl animate-fade-in-up">
        <DialogHeader>
          <DialogTitle className="text-3xl font-display bg-gradient-to-r from-primary to-accent bg-clip-text text-transparent flex items-center gap-2">
            {sign} Horoscope
          </DialogTitle>
        </DialogHeader>
        
        <Tabs defaultValue="daily" className="w-full">
          <TabsList className="grid w-full grid-cols-3 bg-secondary/20">
            <TabsTrigger 
              value="daily"
              className="data-[state=active]:bg-white/80 data-[state=active]:text-primary"
            >
              Daily
            </TabsTrigger>
            <TabsTrigger 
              value="monthly"
              className="data-[state=active]:bg-white/80 data-[state=active]:text-primary"
            >
              Monthly
            </TabsTrigger>
            <TabsTrigger 
              value="traits"
              className="data-[state=active]:bg-white/80 data-[state=active]:text-primary"
            >
              Your Traits
            </TabsTrigger>
          </TabsList>
          
          <TabsContent value="daily" className="mt-4">
            <div className="bg-white/50 p-6 rounded-lg shadow-inner">
              <p className="text-lg leading-relaxed">
                {horoscopeData.daily[sign as keyof typeof horoscopeData.daily]}
              </p>
            </div>
          </TabsContent>
          
          <TabsContent value="monthly" className="mt-4">
            <div className="bg-white/50 p-6 rounded-lg shadow-inner">
              <p className="text-lg leading-relaxed">
                {horoscopeData.monthly[sign as keyof typeof horoscopeData.monthly]}
              </p>
            </div>
          </TabsContent>
          
          <TabsContent value="traits" className="mt-4">
            <div className="space-y-4">
              <div className="bg-white/50 p-6 rounded-lg shadow-inner hover:bg-white/60 transition-colors">
                <h3 className="text-xl font-semibold mb-2 text-primary">Personal Growth</h3>
                <p className="text-gray-700">
                  {horoscopeData.traits[sign as keyof typeof horoscopeData.traits].personal}
                </p>
              </div>
              
              <div className="bg-white/50 p-6 rounded-lg shadow-inner hover:bg-white/60 transition-colors">
                <h3 className="text-xl font-semibold mb-2 text-primary">Relationships</h3>
                <p className="text-gray-700">
                  {horoscopeData.traits[sign as keyof typeof horoscopeData.traits].relationships}
                </p>
              </div>
              
              <div className="bg-white/50 p-6 rounded-lg shadow-inner hover:bg-white/60 transition-colors">
                <h3 className="text-xl font-semibold mb-2 text-primary">Career</h3>
                <p className="text-gray-700">
                  {horoscopeData.traits[sign as keyof typeof horoscopeData.traits].career}
                </p>
              </div>
              
              <div className="bg-white/50 p-6 rounded-lg shadow-inner hover:bg-white/60 transition-colors">
                <h3 className="text-xl font-semibold mb-2 text-primary">Well-being</h3>
                <p className="text-gray-700">
                  {horoscopeData.traits[sign as keyof typeof horoscopeData.traits].wellbeing}
                </p>
              </div>
            </div>
          </TabsContent>
        </Tabs>
      </DialogContent>
    </Dialog>
  );
};

export default HoroscopeModal;