
import React, { useEffect, useState } from 'react';
import Icon from './Icon';
import { getOfferings, purchasePackage, restorePurchases } from '../services/revenueCatService';

interface PaywallModalProps {
  isOpen: boolean;
  onClose: () => void;
  onSuccess: () => void;
}

const PaywallModal: React.FC<PaywallModalProps> = ({ isOpen, onClose, onSuccess }) => {
  const [offering, setOffering] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (isOpen) {
      const load = async () => {
        setLoading(true);
        const off = await getOfferings();
        if (off) setOffering(off);
        else setError("Failed to synchronize with licensing server.");
        setLoading(false);
      };
      load();
    }
  }, [isOpen]);

  if (!isOpen) return null;

  const handlePurchase = async (pkg: any) => {
    setLoading(true);
    try {
      const success = await purchasePackage(pkg);
      if (success) onSuccess();
    } catch (e) {
      alert("Verification failed. Please check network connection.");
    } finally {
      setLoading(false);
    }
  };

  const handleRestore = async () => {
    setLoading(true);
    const success = await restorePurchases();
    if (success) onSuccess();
    else alert("No active subscription found.");
    setLoading(false);
  };

  const packages = offering?.availablePackages || [
    { identifier: 'monthly', packageType: 'MONTHLY', product: { priceString: '$9.99' } },
    { identifier: 'annual', packageType: 'ANNUAL', product: { priceString: '$79.99' } }
  ];

  return (
    <div className="fixed inset-0 z-[200] flex items-center justify-center bg-black/95 backdrop-blur-xl animate-in fade-in duration-300">
      <div className="relative w-full max-w-xl bg-gray-900 border border-white/10 rounded-3xl overflow-hidden shadow-2xl mx-4">
        
        <div className="absolute top-0 left-0 w-full h-1 bg-gradient-to-r from-transparent via-[var(--color-accent)] to-transparent" />
        
        <button onClick={onClose} className="absolute top-6 right-6 p-2 text-gray-500 hover:text-white transition-all z-10">
          <Icon name="x-circle" className="w-8 h-8" />
        </button>

        <div className="p-8 md:p-12 flex flex-col items-center text-center">
          <div className="w-20 h-20 rounded-2xl bg-[var(--color-accent)]/10 flex items-center justify-center mb-8 border border-[var(--color-accent)]/20">
            <Icon name="brain" className="w-10 h-10 text-[var(--color-accent)]" />
          </div>

          <h2 className="text-3xl font-bold tracking-tight mb-2 uppercase text-white">Quadra Seer Intelligence PRO</h2>
          <p className="text-gray-400 mb-10 max-w-sm">Access the full potential of the Seer Core with unhindered predictive power.</p>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 w-full mb-10">
            {packages.map((pkg: any) => (
              <button 
                key={pkg.identifier}
                onClick={() => handlePurchase(pkg)}
                disabled={loading}
                className="group relative flex flex-col p-6 rounded-2xl border border-white/5 bg-white/5 hover:border-[var(--color-accent)]/50 hover:bg-white/10 transition-all text-left"
              >
                <div className="text-[10px] font-bold text-[var(--color-accent)] uppercase mb-1">{pkg.packageType}</div>
                <div className="text-xl font-bold text-white">{pkg.product.priceString}</div>
                <div className="text-[10px] text-gray-500 uppercase mt-auto">Cancel anytime</div>
                {loading && <div className="absolute inset-0 bg-black/40 flex items-center justify-center rounded-2xl animate-pulse" />}
              </button>
            ))}
          </div>

          <ul className="text-left w-full space-y-4 mb-12">
            {[
              "Native Audio Voice Mode",
              "Advanced Logic Derivation",
              "Explorative Predictive Processor",
              "Priority Intelligence Link Access"
            ].map((feature, i) => (
              <li key={i} className="flex items-center gap-3 text-sm text-gray-300">
                <Icon name="sparkles" className="w-4 h-4 text-[var(--color-accent)]" />
                {feature}
              </li>
            ))}
          </ul>

          <div className="flex flex-col gap-4 w-full">
             <button 
              onClick={handleRestore}
              className="text-[10px] text-gray-500 uppercase tracking-widest hover:text-white transition-colors"
            >
              Restore Previous Purchase
            </button>
            <p className="text-[8px] text-gray-700 uppercase leading-relaxed">
              By subscribing, you agree to the Quadra Seer Terms of Service and Privacy Policy. Subscriptions automatically renew unless cancelled.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default PaywallModal;