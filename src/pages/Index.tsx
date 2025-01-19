import React, { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import Header from "@/components/Header";
import { Star, Moon, Sun, Compass, Heart, Sparkles, GemIcon, Brain, Flower2, Infinity, Calendar } from "lucide-react";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Button } from "@/components/ui/button";
import HoroscopeModal from "@/components/HoroscopeModal";
import AstrologyInsightsDialog from "@/components/AstrologyInsightsDialog";
import RemediesDialog from "@/components/RemediesDialog";

const Index = () => {
  const [selectedSign, setSelectedSign] = useState<string | null>(null);
  const [isHoroscopeOpen, setIsHoroscopeOpen] = useState(false);
  const [selectedChakra, setSelectedChakra] = useState<string | null>(null);
  const [isInsightsOpen, setIsInsightsOpen] = useState(false);
  const [isRemediesOpen, setIsRemediesOpen] = useState(false);
  const [chatMessage, setChatMessage] = useState("");

  const zodiacSigns = [
    { name: "Aries", description: "The bold and ambitious", icon: <Star className="text-primary w-6 h-6" /> },
    { name: "Taurus", description: "The steady and reliable", icon: <Moon className="text-primary w-6 h-6" /> },
    { name: "Gemini", description: "The curious and adaptable", icon: <Sun className="text-primary w-6 h-6" /> },
    { name: "Cancer", description: "The nurturing protector", icon: <Heart className="text-primary w-6 h-6" /> },
    { name: "Leo", description: "The confident leader", icon: <Star className="text-primary w-6 h-6" /> },
    { name: "Virgo", description: "The analytical perfectionist", icon: <Brain className="text-primary w-6 h-6" /> },
  ];

  const chakras = [
    { name: "Crown Chakra", color: "violet", description: "Connection to the divine", link: "/chakras/crown" },
    { name: "Third Eye Chakra", color: "indigo", description: "Intuition and foresight", link: "/chakras/third-eye" },
    { name: "Throat Chakra", color: "blue", description: "Communication and truth", link: "/chakras/throat" },
    { name: "Heart Chakra", color: "green", description: "Love and compassion", link: "/chakras/heart" },
    { name: "Solar Plexus Chakra", color: "yellow", description: "Personal power", link: "/chakras/solar-plexus" },
    { name: "Sacral Chakra", color: "orange", description: "Creativity and emotion", link: "/chakras/sacral" },
    { name: "Root Chakra", color: "red", description: "Grounding and stability", link: "/chakras/root" }
  ];

  const spiritualFacts = [
    "Mercury retrograde occurs 3-4 times per year",
    "The moon affects both tides and emotions",
    "Each zodiac sign rules a different part of the body",
    "Your rising sign determines your outward personality",
    "Venus governs love and beauty in astrology"
  ];

  const handleZodiacClick = (sign: string) => {
    setSelectedSign(sign);
    setIsHoroscopeOpen(true);
  };

  const handleChakraClick = (chakraName: string) => {
    setSelectedChakra(chakraName);
  };

  const handleRevealInsights = () => {
    setIsInsightsOpen(true);
  };

  const handleSendMessage = () => {
    if (chatMessage.trim()) {
      setIsRemediesOpen(true);
      setChatMessage("");
    }
  };

  return (
    <div className="min-h-screen">
      <Header />
      
      <main className="container mx-auto px-4 py-8">
        <div className="grid lg:grid-cols-2 gap-8">
          {/* Left Column */}
          <div className="space-y-8">
            {/* Form Container */}
            <Card className="form-container">
              <CardHeader>
                <CardTitle className="text-2xl font-display">Discover Your Spiritual Insights</CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="grid grid-cols-2 gap-4">
                  <div className="space-y-2">
                    <Label htmlFor="year">Year</Label>
                    <Input id="year" type="number" placeholder="YYYY" />
                  </div>
                  <div className="space-y-2">
                    <Label htmlFor="month">Month</Label>
                    <Input id="month" type="number" placeholder="MM" />
                  </div>
                  <div className="space-y-2">
                    <Label htmlFor="day">Day</Label>
                    <Input id="day" type="number" placeholder="DD" />
                  </div>
                  <div className="space-y-2">
                    <Label htmlFor="time">Time</Label>
                    <Input id="time" type="time" />
                  </div>
                </div>
                <div className="space-y-2">
                  <Label htmlFor="location">Location</Label>
                  <Input id="location" type="text" placeholder="Enter your location" />
                </div>
                <Button 
                  className="w-full"
                  onClick={handleRevealInsights}
                >
                  Reveal My Insights
                </Button>
              </CardContent>
            </Card>

            {/* Zodiac Signs */}
            <Card className="bg-white/10 backdrop-blur-lg border-none text-white">
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Star className="text-primary" />
                  Zodiac Signs
                </CardTitle>
              </CardHeader>
              <CardContent className="grid grid-cols-2 gap-4">
                {zodiacSigns.map((sign, index) => (
                  <div
                    key={index}
                    className="p-4 rounded-lg bg-white/5 hover:bg-white/10 transition-colors cursor-pointer"
                    onClick={() => handleZodiacClick(sign.name)}
                  >
                    <div className="flex items-center gap-2 mb-2">
                      {sign.icon}
                      <h3 className="font-semibold">{sign.name}</h3>
                    </div>
                    <p className="text-sm text-gray-300">{sign.description}</p>
                  </div>
                ))}
              </CardContent>
            </Card>

            {/* Spiritual Facts */}
            <Card className="bg-white/10 backdrop-blur-lg border-none text-white">
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Sparkles className="text-primary" />
                  Cosmic Wisdom
                </CardTitle>
              </CardHeader>
              <CardContent>
                <ul className="space-y-4">
                  {spiritualFacts.map((fact, index) => (
                    <li key={index} className="flex items-start gap-3">
                      <Star className="text-primary w-5 h-5 mt-1 flex-shrink-0" />
                      <p>{fact}</p>
                    </li>
                  ))}
                </ul>
              </CardContent>
            </Card>
          </div>

          {/* Right Column */}
          <div className="space-y-8">
            {/* Chatbot */}
            <Card className="chat-container">
              <CardHeader>
                <CardTitle className="text-2xl font-display">Ask SoulBuddy</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="bg-white/5 rounded-lg p-4 mb-4">
                  <p className="text-gray-700">Hello! How can I guide you on your spiritual journey today?</p>
                </div>
                <div className="flex gap-2">
                  <Input 
                    placeholder="Type your message..." 
                    value={chatMessage}
                    onChange={(e) => setChatMessage(e.target.value)}
                    onKeyPress={(e) => {
                      if (e.key === "Enter") {
                        handleSendMessage();
                      }
                    }}
                  />
                  <Button onClick={handleSendMessage}>Send</Button>
                </div>
              </CardContent>
            </Card>

            {/* Chakra System */}
            <Card className="bg-white/10 backdrop-blur-lg border-none text-white">
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Infinity className="text-primary" />
                  Chakra System
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  {chakras.map((chakra, index) => (
                    <Button
                      key={index}
                      className={`w-full p-4 rounded-lg transition-all duration-300 transform hover:scale-105 hover:shadow-lg`}
                      style={{ 
                        backgroundColor: `${chakra.color}20`,
                        borderColor: chakra.color,
                        borderWidth: '1px'
                      }}
                      onClick={() => handleChakraClick(chakra.name)}
                    >
                      <h3 className="font-semibold text-lg mb-2" style={{ color: chakra.color }}>
                        {chakra.name}
                      </h3>
                      <p className="text-sm text-gray-300">{chakra.description}</p>
                    </Button>
                  ))}
                </div>
              </CardContent>
            </Card>

            {/* Home Pooja */}
            <Card className="bg-white/10 backdrop-blur-lg border-none text-white">
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Flower2 className="text-primary" />
                  Home Pooja Services
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <p className="text-gray-300">
                    Experience the divine atmosphere of traditional poojas in the comfort of your home. Our experienced priests perform various rituals according to Vedic traditions.
                  </p>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <Button variant="outline" className="hover:bg-primary/20">
                      <span>Book a Pooja</span>
                    </Button>
                    <Button variant="outline" className="hover:bg-primary/20">
                      <span>View Pooja Types</span>
                    </Button>
                  </div>
                </div>
              </CardContent>
            </Card>

            {/* Religious Events */}
            <Card className="bg-white/10 backdrop-blur-lg border-none text-white">
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Star className="text-primary" />
                  Religious Events
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <p className="text-gray-300">
                    Stay connected with upcoming religious events, festivals, and auspicious dates. Get detailed information about celebrations and their significance.
                  </p>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <Button variant="outline" className="hover:bg-primary/20">
                      <Calendar className="mr-2" />
                      <span>Upcoming Events</span>
                    </Button>
                    <Button variant="outline" className="hover:bg-primary/20">
                      <GemIcon className="mr-2" />
                      <span>Festival Guide</span>
                    </Button>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        </div>
      </main>

      {/* Footer */}
      <footer className="bg-white/5 backdrop-blur-sm mt-16 py-8 text-white">
        <div className="container mx-auto px-4">
          <div className="grid md:grid-cols-3 gap-8">
            <div>
              <h3 className="font-display text-lg mb-4">About SoulBuddy</h3>
              <p className="text-gray-300">
                Your trusted companion on the journey to spiritual enlightenment.
              </p>
            </div>
            <div>
              <h3 className="font-display text-lg mb-4">Quick Links</h3>
              <ul className="space-y-2 text-gray-300">
                <li>
                  <a href="/about" className="hover:text-primary transition-colors">About Us</a>
                </li>
                <li>
                  <a href="/services" className="hover:text-primary transition-colors">Services</a>
                </li>
                <li>
                  <a href="/contact" className="hover:text-primary transition-colors">Contact</a>
                </li>
              </ul>
            </div>
            <div>
              <h3 className="font-display text-lg mb-4">Connect With Us</h3>
              <div className="flex space-x-4">
                <a href="#" className="text-gray-300 hover:text-primary transition-colors">
                  <span className="sr-only">Facebook</span>
                  <Star className="w-6 h-6" />
                </a>
                <a href="#" className="text-gray-300 hover:text-primary transition-colors">
                  <span className="sr-only">Twitter</span>
                  <Moon className="w-6 h-6" />
                </a>
                <a href="#" className="text-gray-300 hover:text-primary transition-colors">
                  <span className="sr-only">Instagram</span>
                  <Sun className="w-6 h-6" />
                </a>
              </div>
            </div>
          </div>
          <div className="mt-8 pt-8 border-t border-white/10 text-center text-gray-300">
            <p>&copy; {new Date().getFullYear()} SoulBuddy. All rights reserved.</p>
          </div>
        </div>
      </footer>

      {/* Modals */}
      {selectedSign && (
        <HoroscopeModal
          isOpen={isHoroscopeOpen}
          onClose={() => setIsHoroscopeOpen(false)}
          sign={selectedSign}
        />
      )}
      
      <AstrologyInsightsDialog
        isOpen={isInsightsOpen}
        onClose={() => setIsInsightsOpen(false)}
      />

      <RemediesDialog
        isOpen={isRemediesOpen}
        onClose={() => setIsRemediesOpen(false)}
      />
    </div>
  );
};

export default Index;
