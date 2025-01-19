import React from "react";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";

interface RemediesDialogProps {
  isOpen: boolean;
  onClose: () => void;
}

const RemediesDialog = ({ isOpen, onClose }: RemediesDialogProps) => {
  return (
    <Dialog open={isOpen} onOpenChange={onClose}>
      <DialogContent className="bg-white/90 text-gray-800 backdrop-blur-lg border border-primary/20 max-w-2xl">
        <DialogHeader>
          <DialogTitle className="text-2xl font-display text-primary">
            Practical Steps and Remedies
          </DialogTitle>
        </DialogHeader>
        
        <div className="space-y-4 mt-4">
          <p className="text-gray-700">
            Here are some commonly recommended remedies for Mangal Dosha:
          </p>
          
          <div className="space-y-6">
            <div>
              <h3 className="text-xl font-semibold text-primary mb-2">1. Marry Another Manglik</h3>
              <p className="text-gray-700">
                It is traditionally believed that when two Manglik individuals marry, the doshas cancel out. This is one of the simplest solutions if both partners agree.
              </p>
            </div>
            
            <div>
              <h3 className="text-xl font-semibold text-primary mb-2">2. Perform Rituals and Pujas</h3>
              <ul className="list-disc pl-5 space-y-2 text-gray-700">
                <li>Conduct a Mangal Shanti Puja: This is a ritual performed to reduce the negative effects of Mangal Dosha.</li>
                <li>Visit temples dedicated to Lord Hanuman or Lord Kartikeya, as they are associated with Mars (Mangal).</li>
                <li>Chant or recite the Hanuman Chalisa daily or on Tuesdays.</li>
                <li>Donate red-colored items like clothes, lentils, or coral to those in need on Tuesdays.</li>
              </ul>
            </div>
            
            <div>
              <h3 className="text-xl font-semibold text-primary mb-2">3. Fasting</h3>
              <p className="text-gray-700">
                Fasting on Tuesdays is considered an effective way to appease Mars. You can consume only fruits or follow a simple satvik diet.
              </p>
            </div>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
};

export default RemediesDialog;