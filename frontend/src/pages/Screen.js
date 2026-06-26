import React, { useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { toast } from "sonner";
import api, { formatApiError } from "@/lib/api";
import { Eye, Hand, Upload, X, Loader2, Sparkles } from "lucide-react";

function fileToBase64(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result.split(",")[1]);
    reader.onerror = reject;
    reader.readAsDataURL(file);
  });
}

function ImageDropzone({ label, hint, image, setImage, testId, icon }) {
  const inputRef = useRef(null);
  const Icon = icon;
  const onPick = (e) => {
    const f = e.target.files?.[0];
    if (!f) return;
    if (!/image\/(jpeg|jpg|png|webp)/i.test(f.type)) {
      toast.error("Please upload JPEG, PNG, or WEBP only");
      return;
    }
    const reader = new FileReader();
    reader.onload = () => setImage({ b64: reader.result.split(",")[1], dataUrl: reader.result, name: f.name });
    reader.readAsDataURL(f);
  };
  return (
    <div className="rounded-2xl border border-[var(--border)] bg-[var(--surface)] p-5">
      <div className="flex items-center gap-2 mb-3">
        <div className="w-8 h-8 rounded-lg bg-[var(--secondary)] grid place-items-center text-[var(--primary)]">
          <Icon className="w-4 h-4" />
        </div>
        <div>
          <div className="font-heading font-medium">{label}</div>
          <div className="text-xs text-[var(--muted)]">{hint}</div>
        </div>
      </div>

      {image ? (
        <div className="relative rounded-xl overflow-hidden border border-[var(--border)]">
          <img src={image.dataUrl} alt={label} className="w-full block" />
          <button
            onClick={() => setImage(null)}
            type="button"
            className="absolute top-2 right-2 bg-black/60 text-white p-1.5 rounded-full hover:bg-black/80"
            data-testid={`${testId}-remove`}
          >
            <X className="w-4 h-4" />
          </button>
        </div>
      ) : (
        <button
          type="button"
          onClick={() => inputRef.current?.click()}
          className="w-full border-2 border-dashed border-[var(--border)] rounded-xl p-8 text-center hover:border-[var(--primary)] hover:bg-[var(--surface-2)] transition"
          data-testid={`${testId}-dropzone`}
        >
          <Upload className="w-6 h-6 mx-auto text-[var(--muted)]" />
          <div className="text-sm mt-2">Click to upload</div>
          <div className="text-xs text-[var(--muted)] mt-1">JPEG · PNG · WEBP</div>
        </button>
      )}
      <input
        ref={inputRef}
        type="file"
        accept="image/jpeg,image/png,image/webp"
        className="hidden"
        onChange={onPick}
        data-testid={`${testId}-input`}
      />
    </div>
  );
}

export default function Screen() {
  const navigate = useNavigate();
  const [eyeImg, setEyeImg] = useState(null);
  const [nailImg, setNailImg] = useState(null);
  const [patientName, setPatientName] = useState("");
  const [patientAge, setPatientAge] = useState("");
  const [patientSex, setPatientSex] = useState("");
  const [notes, setNotes] = useState("");
  const [submitting, setSubmitting] = useState(false);

  const submit = async (e) => {
    e.preventDefault();
    if (!eyeImg && !nailImg) {
      toast.error("Upload at least one image (eye or nail)");
      return;
    }
    setSubmitting(true);
    try {
      const { data } = await api.post("/screenings", {
        eye_image_base64: eyeImg?.b64 || null,
        nail_image_base64: nailImg?.b64 || null,
        patient_name: patientName || null,
        patient_age: patientAge ? parseInt(patientAge, 10) : null,
        patient_sex: patientSex || null,
        notes: notes || null,
      });
      toast.success("Analysis complete");
      navigate(`/results/${data.id}`);
    } catch (e) {
      toast.error(formatApiError(e.response?.data?.detail) || e.message);
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className="max-w-6xl mx-auto px-5 py-10">
      <div className="mb-8 fade-up">
        <div className="text-[10px] uppercase tracking-[0.3em] text-[var(--muted)]">New screening</div>
        <h1 className="font-heading text-4xl sm:text-5xl tracking-tight mt-1">Capture two pictures.</h1>
        <p className="text-[var(--muted)] mt-2 max-w-2xl">
          For best results, take a well-lit photo of the <b>lower eyelid (palpebral conjunctiva)</b> and the <b>fingernail bed</b>.
          The fusion model will combine both signals.
        </p>
      </div>

      <form onSubmit={submit} className="grid lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2 space-y-6 fade-up delay-1">
          <div className="grid md:grid-cols-2 gap-6">
            <ImageDropzone
              label="Eye image"
              hint="Pull the lower eyelid gently down"
              image={eyeImg} setImage={setEyeImg}
              testId="eye" icon={Eye}
            />
            <ImageDropzone
              label="Nail image"
              hint="Place hand flat in good lighting"
              image={nailImg} setImage={setNailImg}
              testId="nail" icon={Hand}
            />
          </div>
          <div className="rounded-2xl border border-[var(--border)] bg-[var(--surface)] p-5">
            <div className="font-heading mb-3">Patient details <span className="text-xs text-[var(--muted)] font-body">(optional)</span></div>
            <div className="grid md:grid-cols-3 gap-3">
              <div>
                <Label htmlFor="pname">Name</Label>
                <Input id="pname" value={patientName} onChange={(e) => setPatientName(e.target.value)} className="rounded-xl mt-1.5" data-testid="patient-name-input" />
              </div>
              <div>
                <Label htmlFor="page">Age</Label>
                <Input id="page" type="number" min="0" max="120" value={patientAge} onChange={(e) => setPatientAge(e.target.value)} className="rounded-xl mt-1.5" data-testid="patient-age-input" />
              </div>
              <div>
                <Label>Sex</Label>
                <Select value={patientSex} onValueChange={setPatientSex}>
                  <SelectTrigger className="rounded-xl mt-1.5" data-testid="patient-sex-select"><SelectValue placeholder="Select" /></SelectTrigger>
                  <SelectContent>
                    <SelectItem value="female">Female</SelectItem>
                    <SelectItem value="male">Male</SelectItem>
                    <SelectItem value="other">Other</SelectItem>
                  </SelectContent>
                </Select>
              </div>
            </div>
            <div className="mt-3">
              <Label htmlFor="notes">Notes</Label>
              <Textarea id="notes" value={notes} onChange={(e) => setNotes(e.target.value)} className="rounded-xl mt-1.5" rows={3} data-testid="patient-notes-input" />
            </div>
          </div>
        </div>

        <aside className="space-y-4 fade-up delay-2">
          <div className="rounded-2xl border border-[var(--border)] bg-[var(--surface)] p-5">
            <div className="flex items-center gap-2 text-[var(--primary)]">
              <Sparkles className="w-4 h-4" />
              <span className="text-[10px] uppercase tracking-[0.25em]">How it works</span>
            </div>
            <h3 className="font-heading text-xl mt-1">What happens when you tap “Run”</h3>
            <ul className="mt-3 space-y-2 text-sm text-[var(--muted)]">
              <li className="flex items-center gap-2"><Eye className="w-4 h-4 text-[var(--primary)]" /> We check your inner eyelid for paleness</li>
              <li className="flex items-center gap-2"><Hand className="w-4 h-4 text-[var(--primary)]" /> We check your nail for paleness &amp; shape</li>
              <li className="flex items-center gap-2"><Sparkles className="w-4 h-4 text-[var(--primary)]" /> We combine both into one easy score</li>
            </ul>
            <Button
              type="submit"
              disabled={submitting || (!eyeImg && !nailImg)}
              className="w-full mt-5 bg-[var(--primary)] hover:bg-[var(--primary-hover)] text-white rounded-xl"
              data-testid="run-analysis-btn"
            >
              {submitting ? (<><Loader2 className="w-4 h-4 mr-2 animate-spin" /> Analyzing…</>) : "Run analysis"}
            </Button>
            <p className="text-[11px] text-[var(--muted)] mt-3 leading-relaxed">
              Results are a screening aid, not a diagnosis. Consult a medical professional for treatment decisions.
            </p>
          </div>
        </aside>
      </form>
    </div>
  );
}
