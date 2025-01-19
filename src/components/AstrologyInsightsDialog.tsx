import React from "react";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";

interface AstrologyInsightsDialogProps {
  isOpen: boolean;
  onClose: () => void;
}

const AstrologyInsightsDialog = ({ isOpen, onClose }: AstrologyInsightsDialogProps) => {
  return (
    <Dialog open={isOpen} onOpenChange={onClose}>
      <DialogContent className="bg-white/90 text-gray-800 backdrop-blur-lg border border-primary/20 max-w-2xl">
        <DialogHeader>
          <DialogTitle className="text-3xl font-display bg-gradient-to-r from-primary to-accent bg-clip-text text-transparent">
            Your Astrological Insights
          </DialogTitle>
        </DialogHeader>
        
        <div className="space-y-4 mt-4">
          <p className="text-lg leading-relaxed">
            Namaste! I'm delighted to help you with your astrological queries. Based on your birth chart, I'll provide you with some insights and guidance.
          </p>
          
          <div className="space-y-4 mt-6">
            <div className="bg-white/50 p-6 rounded-lg shadow-inner">
              <p className="text-gray-700">
                With Jupiter in the 10th house, you're likely to have a strong sense of ambition and a desire for recognition in your career.
              </p>
            </div>
            
            <div className="bg-white/50 p-6 rounded-lg shadow-inner">
              <p className="text-gray-700">
                Saturn in the 6th house can indicate that you may face challenges in your daily routine, health, or work environment.
              </p>
            </div>
            
            <div className="bg-white/50 p-6 rounded-lg shadow-inner">
              <p className="text-gray-700">
                Mars in the 1st house can make you a natural-born leader, with a strong desire to take action and assert yourself in various aspects of life.
              </p>
            </div>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
};

export default AstrologyInsightsDialog;