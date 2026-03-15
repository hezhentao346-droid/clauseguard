"""
ClauseGuard - AI-Powered Contract Risk Analysis
Production-ready backend using Flask + Groq API
"""

import os
import json
import re
import io
import uuid
import hashlib
from datetime import datetime
from flask import Flask, request, jsonify, render_template, send_file, session
from openai import OpenAI
from prompts import SYSTEM_PROMPT, ANALYSIS_PROMPT, QUICK_CLAUSE_PROMPT, EDIT_SUGGESTION_PROMPT

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "clauseguard-dev-key-change-in-prod")
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload

# Groq API setup (OpenAI-compatible, free tier)
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")

client = None
if GROQ_API_KEY:
    client = OpenAI(api_key=GROQ_API_KEY, base_url=GROQ_BASE_URL)

# In-memory storage for contract history (use database in production)
contract_history = {}

LANG_INSTRUCTION = {
    'zh': '\n\nIMPORTANT: You MUST respond with all text fields (risk_explanation, suggested_revision, revision_rationale, executive_summary, negotiation_tips, legal_notes, risk_change, enforceability_notes, legal_reference) in Chinese (Simplified). Keep JSON keys, risk levels (HIGH/MEDIUM/LOW), and party names in English. The original_text should remain as-is from the contract.',
    'en': ''
}


def call_ai(system_prompt, user_prompt, temperature=0.1):
    """Call Groq API with expert contract prompts."""
    if not client:
        return {"error": "GROQ_API_KEY not set. Get one free at https://console.groq.com/keys"}

    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=temperature,
            max_tokens=8000,
            response_format={"type": "json_object"}
        )
        content = response.choices[0].message.content
        return json.loads(content)
    except json.JSONDecodeError:
        try:
            match = re.search(r'\{.*\}', content, re.DOTALL)
            if match:
                return json.loads(match.group())
        except Exception:
            pass
        return {"error": "Failed to parse AI response", "raw": content}
    except Exception as e:
        return {"error": str(e)}


def extract_text_from_file(file):
    """Extract text from uploaded PDF, DOCX, or TXT files."""
    filename = file.filename.lower()

    if filename.endswith('.txt'):
        return file.read().decode('utf-8')

    elif filename.endswith('.pdf'):
        try:
            from PyPDF2 import PdfReader
            reader = PdfReader(io.BytesIO(file.read()))
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
            return text
        except Exception as e:
            return f"Error reading PDF: {str(e)}"

    elif filename.endswith('.docx'):
        try:
            from docx import Document
            doc = Document(io.BytesIO(file.read()))
            return "\n".join([p.text for p in doc.paragraphs])
        except Exception as e:
            return f"Error reading DOCX: {str(e)}"

    return "Unsupported file format. Please upload PDF, DOCX, or TXT."


def get_session_id():
    """Get or create a session ID for tracking contract history."""
    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())
    return session['session_id']


# ============ ROUTES ============

@app.route('/')
def index():
    has_key = bool(GROQ_API_KEY)
    return render_template('index.html', has_api_key=has_key)


@app.route('/api/analyze', methods=['POST'])
def analyze_contract():
    """Full contract risk analysis."""
    contract_text = ""
    lang = 'en'

    if 'file' in request.files and request.files['file'].filename:
        contract_text = extract_text_from_file(request.files['file'])
        lang = request.form.get('lang', 'en')
    elif request.json and 'text' in request.json:
        contract_text = request.json['text']
        lang = request.json.get('lang', 'en')
    else:
        return jsonify({"error": "No contract text provided"}), 400

    if len(contract_text.strip()) < 50:
        return jsonify({"error": "Contract text too short. Please provide a complete contract."}), 400

    prompt = ANALYSIS_PROMPT.replace("{contract_text}", contract_text)
    lang_suffix = LANG_INSTRUCTION.get(lang, '')
    result = call_ai(SYSTEM_PROMPT + lang_suffix, prompt)

    if "error" in result:
        return jsonify(result), 500

    # Save to history
    sid = get_session_id()
    if sid not in contract_history:
        contract_history[sid] = []

    history_entry = {
        "id": str(uuid.uuid4())[:8],
        "timestamp": datetime.utcnow().isoformat(),
        "contract_type": result.get("contract_type", "Unknown"),
        "risk_score": result.get("overall_risk_score", 0),
        "parties": result.get("parties", []),
        "clause_count": len(result.get("clauses", [])),
        "high_risks": len([c for c in result.get("clauses", []) if c.get("risk_level") == "HIGH"]),
        "lang": lang
    }
    contract_history[sid].append(history_entry)

    # Add history ID to result
    result["history_id"] = history_entry["id"]

    return jsonify(result)


@app.route('/api/analyze-clause', methods=['POST'])
def analyze_clause():
    """Analyze a single selected clause."""
    data = request.json
    clause_text = data.get('clause_text', '')
    contract_type = data.get('contract_type', 'General Agreement')
    lang = data.get('lang', 'en')

    if not clause_text:
        return jsonify({"error": "No clause text provided"}), 400

    prompt = QUICK_CLAUSE_PROMPT.replace("{clause_text}", clause_text).replace("{contract_type}", contract_type)
    lang_suffix = LANG_INSTRUCTION.get(lang, '')
    result = call_ai(SYSTEM_PROMPT + lang_suffix, prompt)
    return jsonify(result)


@app.route('/api/suggest-edit', methods=['POST'])
def suggest_edit():
    """Get AI-powered edit suggestion for a clause."""
    data = request.json
    original_text = data.get('original_text', '')
    user_intent = data.get('user_intent', 'Make this clause more balanced and protective')
    contract_type = data.get('contract_type', 'General Agreement')
    lang = data.get('lang', 'en')

    if not original_text:
        return jsonify({"error": "No clause text provided"}), 400

    prompt = (EDIT_SUGGESTION_PROMPT
              .replace("{original_text}", original_text)
              .replace("{user_intent}", user_intent)
              .replace("{contract_type}", contract_type))
    lang_suffix = LANG_INSTRUCTION.get(lang, '')
    result = call_ai(SYSTEM_PROMPT + lang_suffix, prompt)
    return jsonify(result)


@app.route('/api/compare', methods=['POST'])
def compare_clauses():
    """Compare original vs revised clause text."""
    data = request.json
    original = data.get('original', '')
    revised = data.get('revised', '')

    if not original or not revised:
        return jsonify({"error": "Both original and revised text required"}), 400

    # Simple word-level diff
    orig_words = original.split()
    rev_words = revised.split()

    diff_html = []
    import difflib
    matcher = difflib.SequenceMatcher(None, orig_words, rev_words)
    for op, i1, i2, j1, j2 in matcher.get_opcodes():
        if op == 'equal':
            diff_html.append(' '.join(orig_words[i1:i2]))
        elif op == 'delete':
            diff_html.append(f'<del>{" ".join(orig_words[i1:i2])}</del>')
        elif op == 'insert':
            diff_html.append(f'<ins>{" ".join(rev_words[j1:j2])}</ins>')
        elif op == 'replace':
            diff_html.append(f'<del>{" ".join(orig_words[i1:i2])}</del>')
            diff_html.append(f'<ins>{" ".join(rev_words[j1:j2])}</ins>')

    return jsonify({"diff_html": ' '.join(diff_html)})


@app.route('/api/history', methods=['GET'])
def get_history():
    """Get contract analysis history for current session."""
    sid = get_session_id()
    history = contract_history.get(sid, [])
    return jsonify({"history": list(reversed(history))})


@app.route('/api/export', methods=['POST'])
def export_contract():
    """Export the reviewed contract as PDF or DOCX."""
    data = request.json
    clauses = data.get('clauses', [])
    contract_type = data.get('contract_type', 'Reviewed Contract')
    parties = data.get('parties', [])
    export_format = data.get('format', 'pdf')

    if not clauses:
        return jsonify({"error": "No clauses to export"}), 400

    if export_format == 'docx':
        return export_docx(clauses, contract_type, parties)
    else:
        return export_pdf(clauses, contract_type, parties)


def safe_text(text):
    """Remove characters that can't be encoded in latin-1 for PDF."""
    return text.encode('latin-1', errors='replace').decode('latin-1')


def export_pdf(clauses, contract_type, parties):
    """Generate a professional PDF of the reviewed contract."""
    try:
        from fpdf import FPDF

        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=25)
        pdf.add_page()

        pdf.set_font('Helvetica', 'B', 18)
        pdf.cell(0, 15, safe_text(contract_type.upper()), ln=True, align='C')
        pdf.ln(5)

        if parties:
            pdf.set_font('Helvetica', '', 11)
            pdf.cell(0, 8, safe_text(f'Parties: {" and ".join(parties)}'), ln=True, align='C')
            pdf.ln(5)

        pdf.set_font('Helvetica', 'I', 9)
        pdf.set_text_color(100, 100, 100)
        pdf.cell(0, 8, 'Reviewed and revised by ClauseGuard AI', ln=True, align='C')
        pdf.set_text_color(0, 0, 0)
        pdf.ln(10)
        pdf.line(10, pdf.get_y(), 200, pdf.get_y())
        pdf.ln(8)

        for clause in clauses:
            title = clause.get('title', 'Untitled Clause')
            text = clause.get('original_text', '')
            risk = clause.get('risk_level', 'LOW')

            pdf.set_font('Helvetica', 'B', 13)
            risk_label = f' [{risk} RISK]' if risk in ('HIGH', 'MEDIUM') else ''
            pdf.cell(0, 10, safe_text(f'{title}{risk_label}'), ln=True)

            pdf.set_font('Helvetica', '', 11)
            pdf.multi_cell(0, 6, safe_text(text))
            pdf.ln(6)

        pdf.ln(10)
        pdf.line(10, pdf.get_y(), 200, pdf.get_y())
        pdf.ln(5)
        pdf.set_font('Helvetica', 'I', 9)
        pdf.set_text_color(100, 100, 100)
        pdf.cell(0, 8, 'Reviewed by ClauseGuard AI. Not legal advice.', ln=True, align='C')

        buf = io.BytesIO()
        pdf.output(buf)
        buf.seek(0)

        return send_file(buf, mimetype='application/pdf', as_attachment=True,
                         download_name='clauseguard-reviewed-contract.pdf')
    except Exception as e:
        return jsonify({"error": f"PDF export failed: {str(e)}"}), 500


def export_docx(clauses, contract_type, parties):
    """Generate a professional Word document of the reviewed contract."""
    try:
        from docx import Document
        from docx.shared import Pt, RGBColor
        from docx.enum.text import WD_ALIGN_PARAGRAPH
    except ImportError:
        return jsonify({"error": "Word export not available on this server. Use PDF instead."}), 500

    try:
        doc = Document()

        title = doc.add_heading(contract_type.upper(), level=0)
        title.alignment = WD_ALIGN_PARAGRAPH.CENTER

        if parties:
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            run = p.add_run(f'Parties: {" and ".join(parties)}')
            run.font.size = Pt(11)

        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = p.add_run('Reviewed and revised by ClauseGuard AI')
        run.font.size = Pt(9)
        run.font.italic = True
        run.font.color.rgb = RGBColor(128, 128, 128)

        doc.add_paragraph('_' * 80)

        for clause in clauses:
            title_text = clause.get('title', 'Untitled Clause')
            text = clause.get('original_text', '')
            risk = clause.get('risk_level', 'LOW')

            h = doc.add_heading(level=2)
            h.add_run(title_text)
            if risk == 'HIGH':
                risk_run = h.add_run('  [HIGH RISK]')
                risk_run.font.color.rgb = RGBColor(255, 71, 87)
                risk_run.font.size = Pt(10)
            elif risk == 'MEDIUM':
                risk_run = h.add_run('  [MEDIUM RISK]')
                risk_run.font.color.rgb = RGBColor(255, 165, 2)
                risk_run.font.size = Pt(10)

            p = doc.add_paragraph(text)
            p.style.font.size = Pt(11)

        doc.add_paragraph('_' * 80)
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = p.add_run('This document was reviewed using ClauseGuard AI. It does not constitute legal advice.')
        run.font.size = Pt(9)
        run.font.italic = True
        run.font.color.rgb = RGBColor(128, 128, 128)

        buf = io.BytesIO()
        doc.save(buf)
        buf.seek(0)

        return send_file(buf,
                         mimetype='application/vnd.openxmlformats-officedocument.wordprocessingml.document',
                         as_attachment=True,
                         download_name='clauseguard-reviewed-contract.docx')
    except Exception as e:
        return jsonify({"error": f"Word export failed: {str(e)}"}), 500


@app.route('/health')
def health():
    """Health check endpoint for deployment."""
    return jsonify({
        "status": "healthy",
        "model": GROQ_MODEL,
        "api_connected": bool(GROQ_API_KEY)
    })


if __name__ == '__main__':
    if not GROQ_API_KEY:
        print("\n" + "=" * 60)
        print("  WARNING: GROQ_API_KEY not set!")
        print("  Get your free key at: https://console.groq.com/keys")
        print("  Then run: GROQ_API_KEY=your_key python3 app.py")
        print("=" * 60 + "\n")
    else:
        print("\n  ClauseGuard AI is ready!")
        print(f"  Using model: {GROQ_MODEL}\n")

    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, port=port)
