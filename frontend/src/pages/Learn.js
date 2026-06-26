import React from "react";
import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from "@/components/ui/accordion";
import { BookOpen, Salad, HeartPulse, AlertTriangle, Stethoscope } from "lucide-react";

export default function Learn() {
  return (
    <div className="max-w-4xl mx-auto px-5 py-12">
      <div className="fade-up">
        <div className="text-[10px] uppercase tracking-[0.3em] text-[var(--muted)] flex items-center gap-2">
          <BookOpen className="w-3.5 h-3.5" /> Educational
        </div>
        <h1 className="font-heading text-4xl sm:text-5xl tracking-tight mt-1">Understanding anemia</h1>
        <p className="text-[var(--muted)] mt-3 leading-relaxed">
          Anemia is a condition where the blood doesn&apos;t have enough healthy red blood cells or hemoglobin to carry adequate oxygen to the body&apos;s tissues. It is one of the most common nutritional disorders worldwide.
        </p>
      </div>

      <div className="grid md:grid-cols-3 gap-4 mt-8 fade-up delay-1">
        <Card icon={<HeartPulse className="w-5 h-5" />} title="Symptoms" body="Fatigue, weakness, pale skin, shortness of breath, dizziness, cold hands/feet, headaches." />
        <Card icon={<AlertTriangle className="w-5 h-5" />} title="Common causes" body="Iron deficiency, vitamin B12/folate deficiency, blood loss, chronic disease, or genetic conditions." />
        <Card icon={<Stethoscope className="w-5 h-5" />} title="When to consult" body="Persistent fatigue, abnormal pallor, palpitations, or any concerning symptoms warrant a medical evaluation." />
      </div>

      <h2 className="font-heading text-2xl mt-12 mb-3">Diet & prevention</h2>
      <div className="rounded-2xl border border-[var(--border)] bg-[var(--surface)] p-6 fade-up delay-2">
        <div className="flex items-center gap-2 text-[var(--primary)] mb-2"><Salad className="w-4 h-4" /> <span className="text-xs uppercase tracking-[0.25em]">Iron-rich foods</span></div>
        <ul className="grid sm:grid-cols-2 gap-2 text-sm">
          {[
            "Leafy greens (spinach, kale, moringa)",
            "Lentils, chickpeas, beans",
            "Red meat, liver, poultry, fish",
            "Tofu, fortified cereals",
            "Pumpkin & sesame seeds",
            "Pair with vitamin C (citrus, peppers) for better absorption",
          ].map((x, i) => <li key={i} className="flex gap-2"><span className="text-[var(--primary)] mt-1">•</span>{x}</li>)}
        </ul>
      </div>

      <h2 className="font-heading text-2xl mt-12 mb-3">FAQ</h2>
      <Accordion type="single" collapsible className="fade-up delay-3">
        <AccordionItem value="q1" data-testid="faq-q1">
          <AccordionTrigger className="font-heading">How accurate is image-based screening?</AccordionTrigger>
          <AccordionContent className="text-[var(--muted)]">
            Image-based screening is a triage tool. Studies on conjunctival and nail pallor CNNs report ROC-AUC around 0.80–0.90,
            improved further by multimodal fusion. It is not a substitute for a blood test (CBC, hemoglobin).
          </AccordionContent>
        </AccordionItem>
        <AccordionItem value="q2" data-testid="faq-q2">
          <AccordionTrigger className="font-heading">What is Grad-CAM?</AccordionTrigger>
          <AccordionContent className="text-[var(--muted)]">
            Gradient-weighted Class Activation Mapping highlights image regions that most influenced the model&apos;s prediction.
            It makes the model&apos;s decision explainable.
          </AccordionContent>
        </AccordionItem>
        <AccordionItem value="q3" data-testid="faq-q3">
          <AccordionTrigger className="font-heading">Why combine eye + nail?</AccordionTrigger>
          <AccordionContent className="text-[var(--muted)]">
            Each modality captures a complementary signal. Late fusion (weighted by confidence) reduces single-image
            failure modes (bad lighting, makeup, nail polish) and lifts overall accuracy.
          </AccordionContent>
        </AccordionItem>
      </Accordion>

      <p className="text-xs text-[var(--muted)] mt-12 leading-relaxed">
        This information is educational. It is not a substitute for professional medical advice, diagnosis, or treatment.
      </p>
    </div>
  );
}

function Card({ icon, title, body }) {
  return (
    <div className="rounded-2xl border border-[var(--border)] bg-[var(--surface)] p-5">
      <div className="w-9 h-9 rounded-xl bg-[var(--secondary)] grid place-items-center text-[var(--primary)]">{icon}</div>
      <h3 className="font-heading text-lg mt-3">{title}</h3>
      <p className="text-sm text-[var(--muted)] mt-1 leading-relaxed">{body}</p>
    </div>
  );
}
