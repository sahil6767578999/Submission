import React from "react";
import Header from "@/components/Header";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Button } from "@/components/ui/button";
import { useToast } from "@/components/ui/use-toast";

const Contact = () => {
  const { toast } = useToast();

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    toast({
      title: "Message Sent",
      description: "We'll get back to you as soon as possible!",
    });
  };

  return (
    <div className="min-h-screen">
      <Header />
      <main className="container mx-auto px-4 py-8">
        <h1 className="text-4xl font-display text-white mb-6">Contact Us</h1>
        <div className="max-w-2xl mx-auto">
          <form onSubmit={handleSubmit} className="bg-white/80 backdrop-blur-sm rounded-2xl p-8 space-y-6">
            <div className="space-y-2">
              <label htmlFor="name" className="text-gray-700">Name</label>
              <Input id="name" placeholder="Your name" required />
            </div>
            <div className="space-y-2">
              <label htmlFor="email" className="text-gray-700">Email</label>
              <Input id="email" type="email" placeholder="Your email" required />
            </div>
            <div className="space-y-2">
              <label htmlFor="message" className="text-gray-700">Message</label>
              <Textarea id="message" placeholder="Your message" required />
            </div>
            <Button type="submit" className="w-full">Send Message</Button>
          </form>
        </div>
      </main>
    </div>
  );
};

export default Contact;