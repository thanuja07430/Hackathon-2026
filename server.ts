import express, { Request, Response } from 'express';
import dotenv from 'dotenv';
import crypto from 'node:crypto';
import path from 'path';
import fs from 'node:fs';
import multer from 'multer';
import { fileURLToPath } from 'url';
import { GoogleGenAI, Type } from '@google/genai';

dotenv.config();

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
const PORT = process.env.PORT ? parseInt(process.env.PORT, 10) : 3000;

app.use(express.json());

const uploadsRoot = path.join(__dirname, 'uploads');
fs.mkdirSync(uploadsRoot, { recursive: true });
app.use('/uploads', express.static(uploadsRoot));

const allowedAttachmentTypes: Record<string, 'image' | 'video' | 'document'> = {
  'image/jpeg': 'image',
  'image/png': 'image',
  'image/gif': 'image',
  'image/webp': 'image',
  'video/mp4': 'video',
  'video/webm': 'video',
  'video/quicktime': 'video',
  'application/pdf': 'document',
  'application/msword': 'document',
  'application/vnd.openxmlformats-officedocument.wordprocessingml.document': 'document',
  'text/plain': 'document',
};

const attachmentUpload = multer({
  storage: multer.diskStorage({
    destination: (req, _file, callback) => {
      const roomPath = path.join(uploadsRoot, req.params.roomId.toUpperCase());
      fs.mkdirSync(roomPath, { recursive: true });
      callback(null, roomPath);
    },
    filename: (_req, file, callback) => {
      const extension = path.extname(file.originalname).toLowerCase().slice(0, 10);
      callback(null, `${crypto.randomUUID()}${extension}`);
    },
  }),
  limits: { files: 6, fileSize: 50 * 1024 * 1024 },
  fileFilter: (_req, file, callback) => {
    callback(null, Boolean(allowedAttachmentTypes[file.mimetype]));
  },
});

interface UserRecord {
  id: string;
  displayName: string;
  rollNumber: string;
  email: string;
  passwordHash: string;
  studySeconds: number;
}

interface AuthSession {
  token: string;
  userId: string;
  roomCode: string;
  lastSeenAt: number;
}

const users = new Map<string, UserRecord>();
const authSessions = new Map<string, AuthSession>();

const hashPassword = (password: string, salt = crypto.randomBytes(16).toString('hex')) => {
  const hash = crypto.scryptSync(password, salt, 64).toString('hex');
  return `${salt}:${hash}`;
};

const passwordMatches = (password: string, storedHash: string) => {
  const [salt, expectedHash] = storedHash.split(':');
  if (!salt || !expectedHash) return false;
  const actualHash = crypto.scryptSync(password, salt, 64).toString('hex');
  return crypto.timingSafeEqual(Buffer.from(actualHash, 'hex'), Buffer.from(expectedHash, 'hex'));
};

const publicUser = (user: UserRecord) => ({
  id: user.id,
  displayName: user.displayName,
  rollNumber: user.rollNumber,
  email: user.email,
  studySeconds: user.studySeconds,
});

const getAuthSession = (req: Request) => {
  const authorization = req.headers.authorization || '';
  const token = authorization.startsWith('Bearer ') ? authorization.slice(7) : '';
  return authSessions.get(token);
};

const addElapsedStudyTime = (session: AuthSession) => {
  const now = Date.now();
  const elapsedSeconds = Math.min(Math.max(0, now - session.lastSeenAt) / 1000, 120);
  const user = users.get(session.userId);
  if (user) user.studySeconds += elapsedSeconds;
  session.lastSeenAt = now;
};

const findUser = (identifier: string) => {
  const normalized = identifier.trim().toLowerCase();
  return [...users.values()].find(user =>
    user.email.toLowerCase() === normalized || user.rollNumber.toLowerCase() === normalized,
  );
};

const createAuthSession = (user: UserRecord, roomCode = 'ROOM-4092') => {
  const token = crypto.randomBytes(32).toString('hex');
  authSessions.set(token, {
    token,
    userId: user.id,
    roomCode: roomCode.toUpperCase(),
    lastSeenAt: Date.now(),
  });
  return token;
};

const demoUser: UserRecord = {
  id: 'student-demo',
  displayName: 'Alex Morgan',
  rollNumber: 'STU-001',
  email: 'alex@example.com',
  passwordHash: hashPassword('demo123'),
  studySeconds: 0,
};
users.set(demoUser.id, demoUser);

// Initialize Gemini SDK with User-Agent header as required
const ai = new GoogleGenAI({
  apiKey: process.env.GEMINI_API_KEY || '',
  httpOptions: {
    headers: {
      'User-Agent': 'aistudio-build',
    },
  },
});

interface CaptionLine {
  id: string;
  timestamp: string;
  speaker: string;
  text: string;
  highlighted?: boolean;
}

interface LessonAttachment {
  id: string;
  name: string;
  mimeType: string;
  size: number;
  url: string;
  kind: 'image' | 'video' | 'document';
}

interface TeacherLessonPost {
  id: string;
  title: string;
  content: string;
  attachments: LessonAttachment[];
  postedAt: string;
}

interface StudentQuestion {
  id: string;
  studentId: string;
  question: string;
  quickType?: string;
  answer?: string;
  timestamp: string;
  answeredInClass: boolean;
  isPrivate: boolean;
}

interface SimpleNotePoint {
  id: number;
  title: string;
  summary: string;
  bulletDetail: string;
  memoryHook: string;
}

interface VisualCard {
  step: number;
  title: string;
  subtitle: string;
  conceptDescription: string;
  visualCue: string;
  analogy: string;
}

interface QuizOption {
  id: string;
  text: string;
}

type QuizQuestionType = 'multiple_choice' | 'short_answer';

interface QuizQuestion {
  id: string;
  type: QuizQuestionType;
  question: string;
  options?: QuizOption[];
  correctOptionId?: string;
  acceptableKeywords?: string[];
  modelAnswer?: string;
  explanation: string;
  points: number;
}

interface Quiz {
  id: string;
  title: string;
  description: string;
  questions: QuizQuestion[];
  createdAt: string;
  status: 'active' | 'draft' | 'closed';
  totalPoints: number;
}

interface StudentQuizSubmission {
  id: string;
  studentName: string;
  quizId: string;
  score: number;
  maxScore: number;
  percentage: number;
  answers: Record<string, string>;
  questionResults: Record<string, {
    isCorrect: boolean;
    earnedPoints: number;
    feedback: string;
  }>;
  submittedAt: string;
  sharedWithTeacher: boolean;
}

interface ClassroomRoom {
  roomCode: string;
  subject: string;
  grade: string;
  roomNumber: string;
  teacherName: string;
  topic: string;
  status: 'live' | 'paused' | 'ended';
  startedAt: string;
  detectedLanguage?: string;
  captions: CaptionLine[];
  simpleNotes: SimpleNotePoint[];
  visualCards: VisualCard[];
  questions: StudentQuestion[];
  quizzes: Quiz[];
  quizSubmissions: StudentQuizSubmission[];
  teacherNotice?: string;
  lessonPosts?: TeacherLessonPost[];
}

// In-memory persistent state for classroom rooms
const rooms: Record<string, ClassroomRoom> = {
  'ROOM-4092': {
    roomCode: 'ROOM-4092',
    subject: 'Biology',
    grade: 'Grade 8',
    roomNumber: 'Room 4092',
    teacherName: 'Ms. Vance',
    topic: 'Cellular Respiration & Energy Flow',
    status: 'live',
    startedAt: '8:00 AM',
    captions: [
      {
        id: 'c1',
        timestamp: '08:02',
        speaker: 'Ms. Vance',
        text: 'Good morning everyone! Please settle down and open your science notebooks to unit 4.',
      },
      {
        id: 'c2',
        timestamp: '08:03',
        speaker: 'Ms. Vance',
        text: 'Today we are answering a big question: How do living cells turn the food you eat into real energy to jump, think, and breathe?',
      },
      {
        id: 'c3',
        timestamp: '08:04',
        speaker: 'Ms. Vance',
        text: 'In our last class we looked at chloroplasts capturing sunlight in plants. Today we meet their powerhouse partner: the mitochondrion.',
        highlighted: true,
      },
      {
        id: 'c4',
        timestamp: '08:06',
        speaker: 'Ms. Vance',
        text: 'Remember this chemical formula: Glucose plus oxygen yields carbon dioxide, water, and ATP energy.',
      },
      {
        id: 'c5',
        timestamp: '08:07',
        speaker: 'Ms. Vance',
        text: 'Think of ATP as a rechargeable microscopic battery. Every single muscle contraction uses up a tiny burst of ATP.',
        highlighted: true,
      },
      {
        id: 'c6',
        timestamp: '08:09',
        speaker: 'Ms. Vance',
        text: 'Without oxygen, cells can only produce a tiny trickle of energy through anaerobic fermentation, which is why your muscles burn when sprinting.',
      },
    ],
    simpleNotes: [
      {
        id: 1,
        title: 'Mitochondria Are the Powerhouses',
        summary: 'Mitochondria are the organelles where cellular respiration happens in plant and animal cells.',
        bulletDetail: 'They take broken-down sugar and oxygen to make usable chemical fuel.',
        memoryHook: 'Mitochondria = Power plant of the cell.',
      },
      {
        id: 2,
        title: 'The Cellular Respiration Recipe',
        summary: 'Glucose (sugar from food) + Oxygen (from breathing) -> Carbon Dioxide + Water + ATP Energy.',
        bulletDetail: 'Carbon dioxide is exhaled as waste, while ATP is kept to do work.',
        memoryHook: 'In: Food & Air. Out: Energy & Breath.',
      },
      {
        id: 3,
        title: 'ATP Is the Rechargeable Battery',
        summary: 'ATP (Adenosine Triphosphate) stores ready-to-use energy in chemical bonds.',
        bulletDetail: 'When a bond snaps, energy is released for muscle movement, cell repair, and thinking.',
        memoryHook: 'ATP = Fully charged battery ready for action.',
      },
      {
        id: 4,
        title: 'Oxygen Makes It Efficient',
        summary: 'With oxygen (aerobic), one glucose molecule yields around 36 ATP molecules.',
        bulletDetail: 'Without oxygen (anaerobic), cells only produce 2 ATP and lactic acid accumulates.',
        memoryHook: 'Oxygen gives 18x more fuel than sprint mode.',
      },
    ],
    visualCards: [
      {
        step: 1,
        title: 'Food & Oxygen Enter the Cell',
        subtitle: 'Raw Materials Intake',
        conceptDescription: 'Glucose molecules from digested nutrition and oxygen from the lungs diffuse across the cell membrane into the cytoplasm.',
        visualCue: 'Glucose hexagon (green) and O2 pairs (blue) crossing the cell boundary.',
        analogy: 'Delivering raw timber and fire starter to a workshop.',
      },
      {
        step: 2,
        title: 'Glycolysis: The First Split',
        subtitle: 'Cytoplasm Breakdown',
        conceptDescription: 'Inside the cell fluid, the 6-carbon glucose is split in half into two 3-carbon pyruvate pieces. This creates a quick 2 ATP.',
        visualCue: 'A hexagon snapping into two equal triangles with two glowing energy sparks.',
        analogy: 'Chopping a long log into two manageable firewood sticks.',
      },
      {
        step: 3,
        title: 'Mitochondria Furnace Fires Up',
        subtitle: 'Inner Membrane Cycle',
        conceptDescription: 'The split pieces enter the folded inner cristae of the mitochondrion. Oxygen is consumed in the electron transport chain.',
        visualCue: 'Intricate maze-like mitochondrial folds glowing warm amber with swirling electrons.',
        analogy: 'Feeding fuel into a super-efficient furnace with active ventilation.',
      },
      {
        step: 4,
        title: 'Massive ATP Battery Charge',
        subtitle: 'Energy Release & Byproducts',
        conceptDescription: 'Around 30-34 additional ATP batteries are charged. Clean water and carbon dioxide are produced as harmless byproducts.',
        visualCue: 'Row of vibrant batteries turning from depleted gray to bright glowing teal.',
        analogy: 'Recharging all smartphones and tools in the station simultaneously.',
      },
    ],
    questions: [
      {
        id: 'q1',
        studentId: 'Student · Desk 4',
        question: 'Why do plants need mitochondria if they already have chloroplasts?',
        quickType: 'Give me an example',
        answer: 'Chloroplasts make sugar from sunlight, but plants still need mitochondria to break that sugar down into ATP energy at night or in roots deep underground where there is no sun!',
        timestamp: '08:08',
        answeredInClass: true,
        isPrivate: false,
      },
    ],
    quizzes: [
      {
        id: 'quiz-bio-1',
        title: 'Cellular Respiration & ATP Mastery Check',
        description: 'Test your understanding of mitochondria, chemical formulas, and ATP energy conversion.',
        status: 'active',
        createdAt: '08:10 AM',
        totalPoints: 10,
        questions: [
          {
            id: 'q-bio-1',
            type: 'multiple_choice',
            question: 'Which cellular organelle is nicknamed the "powerhouse of the cell" for generating usable chemical fuel?',
            options: [
              { id: 'opt-1', text: 'Ribosome' },
              { id: 'opt-2', text: 'Mitochondrion' },
              { id: 'opt-3', text: 'Nucleus' },
              { id: 'opt-4', text: 'Endoplasmic Reticulum' },
            ],
            correctOptionId: 'opt-2',
            explanation: 'Mitochondria are the primary site of aerobic cellular respiration where glucose and oxygen are converted into ATP.',
            points: 2,
          },
          {
            id: 'q-bio-2',
            type: 'multiple_choice',
            question: 'What are the two raw materials required as inputs for aerobic cellular respiration?',
            options: [
              { id: 'opt-1', text: 'Carbon dioxide and Water' },
              { id: 'opt-2', text: 'Nitrogen and Sunlight' },
              { id: 'opt-3', text: 'Glucose and Oxygen' },
              { id: 'opt-4', text: 'Lactic acid and ATP' },
            ],
            correctOptionId: 'opt-3',
            explanation: 'The inputs are Glucose (from food) and Oxygen (from respiration). The outputs are Carbon dioxide, Water, and ATP.',
            points: 2,
          },
          {
            id: 'q-bio-3',
            type: 'short_answer',
            question: 'In your own words, why is ATP compared to a "rechargeable battery" in living cells?',
            acceptableKeywords: ['store', 'energy', 'charge', 'release', 'bond', 'battery', 'power', 'usable'],
            modelAnswer: 'ATP stores ready-to-use energy in chemical bonds and releases a burst of power when snapped, then can be recharged again through cellular respiration.',
            explanation: 'ATP can be repeatedly charged with energy from food and discharged whenever the cell needs to perform mechanical or chemical work.',
            points: 3,
          },
          {
            id: 'q-bio-4',
            type: 'multiple_choice',
            question: 'Why do your muscles burn during an all-out sprint when oxygen levels drop?',
            options: [
              { id: 'opt-1', text: 'Cells switch to anaerobic fermentation producing lactic acid with only 2 ATP' },
              { id: 'opt-2', text: 'The mitochondria explode from excess heat' },
              { id: 'opt-3', text: 'Cells run out of water completely' },
              { id: 'opt-4', text: 'Chloroplasts start operating in muscle cells' },
            ],
            correctOptionId: 'opt-1',
            explanation: 'Without oxygen, cells cannot complete the mitochondrial electron transport chain and must rely on fermentation, which produces lactic acid.',
            points: 3,
          },
        ],
      },
    ],
    quizSubmissions: [
      {
        id: 'sub-sample-1',
        studentName: 'Alex M. (Desk 2)',
        quizId: 'quiz-bio-1',
        score: 10,
        maxScore: 10,
        percentage: 100,
        answers: {
          'q-bio-1': 'opt-2',
          'q-bio-2': 'opt-3',
          'q-bio-3': 'ATP stores energy like a charged battery and releases it for muscles to move.',
          'q-bio-4': 'opt-1',
        },
        questionResults: {
          'q-bio-1': { isCorrect: true, earnedPoints: 2, feedback: 'Correct! Mitochondrion is the powerhouse.' },
          'q-bio-2': { isCorrect: true, earnedPoints: 2, feedback: 'Correct! Glucose and oxygen are the inputs.' },
          'q-bio-3': { isCorrect: true, earnedPoints: 3, feedback: 'Excellent explanation of storing and releasing energy.' },
          'q-bio-4': { isCorrect: true, earnedPoints: 3, feedback: 'Correct! Fermentation produces lactic acid and 2 ATP.' },
        },
        submittedAt: '08:12 AM',
        sharedWithTeacher: true,
      },
    ],
    teacherNotice: 'Remember to complete the quick 4-question check-in at the end of class.',
  },
  'ROOM-104': {
    roomCode: 'ROOM-104',
    subject: 'World History',
    grade: 'Grade 9',
    roomNumber: 'Room 104',
    teacherName: 'Mr. Harrison',
    topic: 'The Silk Road & Cultural Exchange',
    status: 'live',
    startedAt: '8:15 AM',
    captions: [
      {
        id: 'hc1',
        timestamp: '08:16',
        speaker: 'Mr. Harrison',
        text: 'Welcome Grade 9. Today we demystify the Silk Road.',
      },
      {
        id: 'hc2',
        timestamp: '08:18',
        speaker: 'Mr. Harrison',
        text: 'The most important thing to know is that it was not a single paved highway, but an expansive web of caravan routes.',
      },
      {
        id: 'hc3',
        timestamp: '08:20',
        speaker: 'Mr. Harrison',
        text: 'While luxury goods like silk, spices, and porcelain traveled west, ideas, religions, and printing technologies also spread.',
        highlighted: true,
      },
    ],
    simpleNotes: [
      {
        id: 1,
        title: 'Not One Road, But a Network',
        summary: 'The Silk Road was an overland and maritime web linking East Asia, the Middle East, and the Mediterranean.',
        bulletDetail: 'Merchants rarely traveled the entire length; goods moved hand-to-hand across oasis trading posts.',
        memoryHook: 'Like an ancient relay race across continents.',
      },
      {
        id: 2,
        title: 'High Value, Low Weight Goods',
        summary: 'Silk, jade, spices, glass, and paper were the primary cargo because long-distance caravan travel was costly.',
        bulletDetail: 'Bulky items were saved for maritime ships along coastal routes.',
        memoryHook: 'Only lightweight treasures were worth the journey.',
      },
      {
        id: 3,
        title: 'Ideas Traveled With the Goods',
        summary: 'Buddhism, Islam, paper making, and astronomy spread across cultures along caravan trails.',
        bulletDetail: 'Cultural exchange had far longer lasting impact than the silk itself.',
        memoryHook: 'The real wealth exchanged was human knowledge.',
      },
      {
        id: 4,
        title: 'Caravanserais Were Safe Havens',
        summary: 'Fortified roadside inns (caravanserais) provided water, food, and security from bandits for travelers and camels.',
        bulletDetail: 'They served as melting pots for language and currency exchange.',
        memoryHook: 'Ancient travel hubs with rooms, stable, and markets.',
      },
    ],
    visualCards: [
      {
        step: 1,
        title: 'Production of Rare Commodities',
        subtitle: 'Silk Workshops & Spice Groves',
        conceptDescription: 'Chinese silk weavers cultivate silkworms while Indian and Southeast Asian islands harvest nutmeg and pepper.',
        visualCue: 'Spools of gleaming raw gold/ivory silk threads and dried spice sacks.',
        analogy: 'Developing exclusive luxury technology in regional centers.',
      },
      {
        step: 2,
        title: 'Caravan Assembly at Oasis Hubs',
        subtitle: 'Dunhuang and Samarkand',
        conceptDescription: 'Merchants form resilient camel caravans with armed guides to cross the perilous Taklamakan Desert.',
        visualCue: 'Line of Bactrian camels loaded with woven packs under desert dunes.',
        analogy: 'Organizing freight convoys before traversing hostile territory.',
      },
      {
        step: 3,
        title: 'Trading at the Caravanserai',
        subtitle: 'Exchange & Translation',
        conceptDescription: 'Every 20 to 30 miles, caravans rest inside walled compounds where goods are bartered and languages blend.',
        visualCue: 'Archway courtyard with merchants negotiating over scrolls and scales.',
        analogy: 'International airport lounge where ideas and currencies mix.',
      },
      {
        step: 4,
        title: 'Arrival in Mediterranean Ports',
        subtitle: 'Venice, Alexandria & Constantinople',
        conceptDescription: 'Goods reach Mediterranean ports where prices multiply twenty-fold, transforming European fashion and science.',
        visualCue: 'Sailing ships docking at bustling Roman/Byzantine stone quays.',
        analogy: 'Imported flagship devices hitting retail shelves with immense demand.',
      },
    ],
    questions: [],
    quizzes: [
      {
        id: 'quiz-history-1',
        title: 'Silk Road Commerce & Culture Check',
        description: 'Verify your understanding of Eurasian trade networks, goods, and cultural exchange.',
        status: 'active',
        createdAt: '08:22 AM',
        totalPoints: 8,
        questions: [
          {
            id: 'q-hist-1',
            type: 'multiple_choice',
            question: 'What was the primary characteristic of the Silk Road routes?',
            options: [
              { id: 'opt-1', text: 'A single paved superhighway built by one dynasty' },
              { id: 'opt-2', text: 'An expansive web of overland and maritime relay routes' },
              { id: 'opt-3', text: 'An exclusively underwater canal system' },
              { id: 'opt-4', text: 'A railroad network connecting kingdoms' },
            ],
            correctOptionId: 'opt-2',
            explanation: 'The Silk Road was never one singular road, but a dynamic network of caravan trails and maritime routes.',
            points: 2,
          },
          {
            id: 'q-hist-2',
            type: 'short_answer',
            question: 'Besides precious commodities like silk and porcelain, name two important things that spread along the Silk Road.',
            acceptableKeywords: ['religion', 'idea', 'culture', 'buddhism', 'islam', 'paper', 'technology', 'astronomy', 'disease', 'philosophy'],
            modelAnswer: 'Religions (such as Buddhism and Islam), technologies (like paper-making), and scientific ideas spread along the network.',
            explanation: 'The transfer of intellectual, religious, and technical knowledge had a greater long-term global impact than the luxury goods.',
            points: 3,
          },
          {
            id: 'q-hist-3',
            type: 'multiple_choice',
            question: 'What was the key function of roadside caravanserais along desert paths?',
            options: [
              { id: 'opt-1', text: 'They served as secure fortified inns with water, lodging, and market stalls for traders' },
              { id: 'opt-2', text: 'They were military prisons for captured merchants' },
              { id: 'opt-3', text: 'They were factories where all silk was spun' },
              { id: 'opt-4', text: 'They were lighthouses for guiding desert ships' },
            ],
            correctOptionId: 'opt-1',
            explanation: 'Caravanserais were essential resting nodes spaced about 20-30 miles apart that provided safety and provisions.',
            points: 3,
          },
        ],
      },
    ],
    quizSubmissions: [],
    teacherNotice: 'Map assignment due on Thursday at 3 PM.',
  },
};

// Helper: Ensure room exists or seed on the fly
function getOrCreateRoom(roomCode: string): ClassroomRoom {
  const code = roomCode.toUpperCase().trim();
  if (!rooms[code]) {
    rooms[code] = {
      roomCode: code,
      subject: 'General Classroom',
      grade: 'All Grades',
      roomNumber: code,
      teacherName: 'Teacher',
      topic: 'Live Accessible Classroom Session',
      status: 'live',
      startedAt: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
      captions: [
        {
          id: 'init-1',
          timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
          speaker: 'Teacher',
          text: `Welcome to classroom room ${code}. The text bridge is now connected and listening.`,
        },
      ],
      simpleNotes: [
        {
          id: 1,
          title: 'Classroom Text Bridge Connected',
          summary: `You are connected to ${code}. Real-time transcription and note simplification are active.`,
          bulletDetail: 'Captions generated in this room will automatically synthesize into 4 core points.',
          memoryHook: 'Calm, clear focus is ready.',
        },
        {
          id: 2,
          title: 'Ask Questions Discreetly',
          summary: 'Use the student input box below to ask any clarifying question without interrupting.',
          bulletDetail: 'Quick actions let you ask for instant examples or simple definitions.',
          memoryHook: 'Never hesitate to clarify.',
        },
        {
          id: 3,
          title: 'Visual Learning Sequence Available',
          summary: 'Complex lessons can be viewed as sequential visual step cards.',
          bulletDetail: 'Switch tabs anytime to see visual steps and analogies.',
          memoryHook: 'See the concepts step by step.',
        },
        {
          id: 4,
          title: 'Exam Helper Ready',
          summary: 'Paste any complex test question or instruction to break it down into clean steps.',
          bulletDetail: 'Eliminates confusing filler words and highlights action verbs.',
          memoryHook: 'Clear checklist for every task.',
        },
      ],
      visualCards: [
        {
          step: 1,
          title: 'Join the Classroom Room',
          subtitle: 'Active Room Link',
          conceptDescription: `Students and teachers join ${code} to share live transcriptions and interactive notes.`,
          visualCue: 'Digital bridge connecting classroom desk to accessible display.',
          analogy: 'Tuning into the clear classroom radio channel.',
        },
      ],
      questions: [],
      quizzes: [],
      quizSubmissions: [],
      teacherNotice: 'Room created. Teacher mic is active.',
    };
  }
  return rooms[code];
}

app.post('/api/auth/register', (req: Request, res: Response) => {
  const { displayName, rollNumber, email, password } = req.body;
  if (!displayName?.trim() || !rollNumber?.trim() || !email?.trim() || typeof password !== 'string' || password.length < 6) {
    return res.status(400).json({ error: 'Name, roll number, email, and a password of 6+ characters are required.' });
  }
  if (findUser(email) || findUser(rollNumber)) {
    return res.status(409).json({ error: 'That email or roll number is already registered.' });
  }

  const user: UserRecord = {
    id: crypto.randomUUID(),
    displayName: displayName.trim(),
    rollNumber: rollNumber.trim().toUpperCase(),
    email: email.trim().toLowerCase(),
    passwordHash: hashPassword(password),
    studySeconds: 0,
  };
  users.set(user.id, user);
  res.json({ success: true, user: publicUser(user), token: createAuthSession(user) });
});

app.post('/api/auth/login', (req: Request, res: Response) => {
  const { identifier, password } = req.body;
  const user = typeof identifier === 'string' ? findUser(identifier) : undefined;
  if (!user || typeof password !== 'string' || !passwordMatches(password, user.passwordHash)) {
    return res.status(401).json({ error: 'We could not match that login. Check your roll number/email and password.' });
  }
  res.json({ success: true, user: publicUser(user), token: createAuthSession(user) });
});

app.get('/api/auth/me', (req: Request, res: Response) => {
  const session = getAuthSession(req);
  const user = session ? users.get(session.userId) : undefined;
  if (!session || !user) return res.status(401).json({ error: 'Session expired.' });
  addElapsedStudyTime(session);
  res.json({ success: true, user: publicUser(user) });
});

app.post('/api/auth/heartbeat', (req: Request, res: Response) => {
  const session = getAuthSession(req);
  const user = session ? users.get(session.userId) : undefined;
  if (!session || !user) return res.status(401).json({ error: 'Session expired.' });
  addElapsedStudyTime(session);
  if (typeof req.body.roomCode === 'string') session.roomCode = req.body.roomCode.toUpperCase();
  res.json({ success: true, user: publicUser(user) });
});

app.post('/api/auth/logout', (req: Request, res: Response) => {
  const session = getAuthSession(req);
  if (session) {
    addElapsedStudyTime(session);
    authSessions.delete(session.token);
  }
  res.json({ success: true });
});

// 1. Get room data
app.get('/api/rooms/:roomId', (req: Request, res: Response) => {
  const room = getOrCreateRoom(req.params.roomId);
  res.json(room);
});

app.put('/api/rooms/:roomId/notes', (req: Request, res: Response) => {
  const room = getOrCreateRoom(req.params.roomId);
  const notes = req.body?.notes;
  if (!Array.isArray(notes) || notes.length === 0 || notes.length > 20) {
    return res.status(400).json({ error: 'Provide between 1 and 20 lesson notes.' });
  }

  const validNotes = notes.filter((note: any) =>
    Number.isFinite(Number(note?.id)) &&
    typeof note?.title === 'string' && note.title.trim() &&
    typeof note?.summary === 'string' && note.summary.trim() &&
    typeof note?.bulletDetail === 'string' &&
    typeof note?.memoryHook === 'string',
  ).map((note: any) => ({
    id: Number(note.id),
    title: note.title.trim(),
    summary: note.summary.trim(),
    bulletDetail: note.bulletDetail.trim(),
    memoryHook: note.memoryHook.trim(),
  }));

  if (validNotes.length !== notes.length) {
    return res.status(400).json({ error: 'Each note needs a title, summary, detail, and memory hook.' });
  }

  room.simpleNotes = validNotes;
  res.json({ success: true, notes: room.simpleNotes });
});

app.post('/api/rooms/:roomId/lesson-attachments', attachmentUpload.array('files', 6), (req: Request, res: Response) => {
  const files = (req.files || []) as Express.Multer.File[];
  if (files.length === 0) return res.status(400).json({ error: 'Choose at least one supported file.' });

  const roomCode = req.params.roomId.toUpperCase();
  const attachments: LessonAttachment[] = files.map(file => ({
    id: path.basename(file.filename, path.extname(file.filename)),
    name: file.originalname,
    mimeType: file.mimetype,
    size: file.size,
    url: `/uploads/${encodeURIComponent(roomCode)}/${encodeURIComponent(file.filename)}`,
    kind: allowedAttachmentTypes[file.mimetype],
  }));
  res.json({ success: true, attachments });
});

app.post('/api/rooms/:roomId/lesson-posts', (req: Request, res: Response) => {
  const room = getOrCreateRoom(req.params.roomId);
  const { title, content, attachments = [] } = req.body || {};
  if (typeof title !== 'string' || !title.trim() || typeof content !== 'string' || !content.trim()) {
    return res.status(400).json({ error: 'Lesson title and content are required.' });
  }
  if (!Array.isArray(attachments) || attachments.length > 6 || attachments.some((attachment: any) => !attachment?.id || !attachment?.url || !attachment?.kind)) {
    return res.status(400).json({ error: 'Invalid lesson attachments.' });
  }

  const post: TeacherLessonPost = {
    id: `lesson-${Date.now()}-${Math.random().toString(36).slice(2, 7)}`,
    title: title.trim(),
    content: content.trim(),
    attachments,
    postedAt: new Date().toLocaleString(),
  };
  room.lessonPosts = [post, ...(room.lessonPosts || [])];
  res.json({ success: true, post });
});

// 2. Add caption line
app.post('/api/rooms/:roomId/caption', (req: Request, res: Response) => {
  const room = getOrCreateRoom(req.params.roomId);
  const { text, speaker = room.teacherName, highlighted = false } = req.body;

  if (!text || typeof text !== 'string') {
    res.status(400).json({ error: 'Caption text is required' });
    return;
  }

  const newCaption: CaptionLine = {
    id: `c-${Date.now()}-${Math.random().toString(36).slice(2, 6)}`,
    timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
    speaker,
    text: text.trim(),
    highlighted: Boolean(highlighted),
  };

  room.captions.push(newCaption);
  res.json({ success: true, caption: newCaption, totalCaptions: room.captions.length });
});

// 3. Ask question without interrupting
app.post('/api/rooms/:roomId/question', async (req: Request, res: Response) => {
  const room = getOrCreateRoom(req.params.roomId);
  const { question, quickType, captionContext, studentName = 'Quiet Student' } = req.body;

  if (!question || typeof question !== 'string') {
    res.status(400).json({ error: 'Question text is required' });
    return;
  }

  const selectedCaption = typeof captionContext === 'string' ? captionContext.trim() : '';

  const qId = `q-${Date.now()}`;
  let generatedAnswer = '';

  // Gather recent classroom context (last 8 captions)
  const recentCaptions = room.captions.slice(-10).map(c => `${c.speaker}: ${c.text}`).join('\n');

  try {
    if (process.env.GEMINI_API_KEY) {
      const prompt = `You are "ClassEase Text Bridge", an empathetic, gentle, and hyper-clear classroom learning companion for students.
The student is sitting in class (${room.subject} - ${room.topic}).
The teacher recently said:
"""
${recentCaptions || 'Lesson just started.'}
"""

The student asks quietly without interrupting the teacher:
"${question}"
${selectedCaption ? `
Selected classroom statement to explain directly:
"""
${selectedCaption}
"""` : ''}
Special request type: ${quickType || 'Standard question'}

Rules:
1. Provide a calm, reassuring, direct answer in 2 to 4 friendly sentences.
2. Use clear, accessible language suitable for ${room.grade || 'middle school'}.
3. If they asked "Repeat the last point", distill what the teacher just said in plain words.
4. If they asked "Give me an example", provide a concrete, vivid real-world analogy.
5. If they asked "Explain like I'm 10", use simple analogies and zero jargon.
6. Do not mention that you are an AI model. Speak as the friendly classroom text bridge.`;

      const response = await ai.models.generateContent({
        model: 'gemini-3.8-flash',
        contents: prompt,
      });

      generatedAnswer = response.text?.trim() || '';
    }
  } catch (err) {
    console.error('Gemini question answering error:', err);
  }

  if (!generatedAnswer) {
    // Intelligent contextual fallback
    if (quickType === 'Explain this caption' && selectedCaption) {
      generatedAnswer = `In simple words, the teacher means: "${selectedCaption}" This statement connects to ${room.topic} because it highlights one important part of how the lesson works.`;
    } else if (quickType === 'Repeat the last point' || question.toLowerCase().includes('repeat')) {
      const lastLine = room.captions[room.captions.length - 1]?.text || 'The teacher is introducing the main topic.';
      generatedAnswer = `In simple terms: The teacher emphasized that "${lastLine}". The key idea is how this directly connects to the lesson's main goal.`;
    } else if (quickType === 'Give me an example' || question.toLowerCase().includes('example')) {
      generatedAnswer = `Here is a helpful everyday example: Think of it like a rechargeable power bank. When you charge it with food and air, it holds that power until you need to run your apps or sprint!`;
    } else if (quickType === "Explain like I'm 10") {
      generatedAnswer = `Imagine your body is a giant toy city. The cells are tiny factories inside, and they need fuel (sugar) and fresh air (oxygen) to turn on the lights and keep the machines humming.`;
    } else {
      generatedAnswer = `Great question. In this lesson on ${room.topic}, remember that every main process happens in sequential steps so that the cell or system stays balanced and energized.`;
    }
  }

  const studentQuestion: StudentQuestion = {
    id: qId,
    studentId: studentName,
    question: question.trim(),
    quickType,
    answer: generatedAnswer,
    timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
    answeredInClass: false,
    isPrivate: false,
  };

  room.questions.unshift(studentQuestion);
  res.json({ success: true, question: studentQuestion });
});

// 4. Generate Simple Notes (Auto-generated 4 key points)
app.post('/api/rooms/:roomId/generate-notes', async (req: Request, res: Response) => {
  const room = getOrCreateRoom(req.params.roomId);
  const transcriptText = room.captions.map(c => `${c.speaker}: ${c.text}`).join('\n');

  try {
    if (process.env.GEMINI_API_KEY && transcriptText.length > 30) {
      const prompt = `You are ClassEase. Analyze this live classroom transcript from ${room.subject} (${room.grade}, Topic: ${room.topic}):
"""
${transcriptText}
"""

Generate EXACTLY 4 simple, highly accessible key points for a student with ADHD or auditory processing needs.
Return valid JSON matching this schema:
[
  {
    "id": 1,
    "title": "Short Punchy Title",
    "summary": "One clear sentence explaining the core concept.",
    "bulletDetail": "Supporting detail or how it works.",
    "memoryHook": "A simple 4-6 word memory hook or analogy."
  }
]`;

      const response = await ai.models.generateContent({
        model: 'gemini-3.8-flash',
        contents: prompt,
        config: {
          responseMimeType: 'application/json',
          responseSchema: {
            type: Type.ARRAY,
            items: {
              type: Type.OBJECT,
              properties: {
                id: { type: Type.INTEGER },
                title: { type: Type.STRING },
                summary: { type: Type.STRING },
                bulletDetail: { type: Type.STRING },
                memoryHook: { type: Type.STRING },
              },
              required: ['id', 'title', 'summary', 'bulletDetail', 'memoryHook'],
            },
          },
        },
      });

      const parsed = JSON.parse(response.text?.trim() || '[]');
      if (Array.isArray(parsed) && parsed.length > 0) {
        room.simpleNotes = parsed.slice(0, 4);
        res.json({ success: true, notes: room.simpleNotes });
        return;
      }
    }
  } catch (err) {
    console.error('Gemini generate-notes error:', err);
  }

  // Fallback to room's existing or structured synthesized notes
  res.json({ success: true, notes: room.simpleNotes });
});

// 5. Generate Visual Learning Cards
app.post('/api/rooms/:roomId/visual-cards', async (req: Request, res: Response) => {
  const room = getOrCreateRoom(req.params.roomId);
  const transcriptText = room.captions.map(c => `${c.speaker}: ${c.text}`).join('\n');

  try {
    if (process.env.GEMINI_API_KEY && transcriptText.length > 30) {
      const prompt = `You are ClassEase Visual Learning Designer.
Convert this classroom lesson (${room.subject}: ${room.topic}) into a sequence of 4 sequential visual learning cards:
"""
${transcriptText}
"""

Return valid JSON with 4 items:
[
  {
    "step": 1,
    "title": "Stage Name",
    "subtitle": "Clear Phase Subtitle",
    "conceptDescription": "Clear plain language explanation of what happens in this step.",
    "visualCue": "Description of the visual diagram or shape to picture.",
    "analogy": "Memorable real-life analogy."
  }
]`;

      const response = await ai.models.generateContent({
        model: 'gemini-3.8-flash',
        contents: prompt,
        config: {
          responseMimeType: 'application/json',
          responseSchema: {
            type: Type.ARRAY,
            items: {
              type: Type.OBJECT,
              properties: {
                step: { type: Type.INTEGER },
                title: { type: Type.STRING },
                subtitle: { type: Type.STRING },
                conceptDescription: { type: Type.STRING },
                visualCue: { type: Type.STRING },
                analogy: { type: Type.STRING },
              },
              required: ['step', 'title', 'subtitle', 'conceptDescription', 'visualCue', 'analogy'],
            },
          },
        },
      });

      const parsed = JSON.parse(response.text?.trim() || '[]');
      if (Array.isArray(parsed) && parsed.length > 0) {
        room.visualCards = parsed.slice(0, 4);
        res.json({ success: true, cards: room.visualCards });
        return;
      }
    }
  } catch (err) {
    console.error('Gemini visual-cards error:', err);
  }

  res.json({ success: true, cards: room.visualCards });
});

// 6. Accessible Exam Helper endpoint
app.post('/api/exam-helper', async (req: Request, res: Response) => {
  const { promptText, gradeLevel = 'Grade 8' } = req.body;

  if (!promptText || typeof promptText !== 'string') {
    res.status(400).json({ error: 'Exam question text is required' });
    return;
  }

  try {
    if (process.env.GEMINI_API_KEY) {
      const systemInstruction = `You are the ClassEase Accessible Exam Helper.
Your mission is to support neurodivergent students, students with anxiety, ESL students, and all learners by translating confusing, bloated exam questions or assignment instructions into crystal-clear actionable steps.
Remove double negatives, confusing academic filler, and passive voice.
Identify imperative action verbs (e.g., Explain, Compare, Calculate, State, Justify).
Break the instructions into a clean step-by-step checklist.
Provide short definitions for difficult technical terms.`;

      const userContent = `Convert this exam prompt for ${gradeLevel}:
"""
${promptText}
"""

Return JSON matching:
{
  "cleanedPrompt": "The question rewritten in direct, friendly, plain language with zero fluff.",
  "actionVerbs": ["Action verb 1", "Action verb 2"],
  "clearSteps": [
    {
      "stepNumber": 1,
      "action": "What to do first",
      "hint": "Helpful tip on what the grader is looking for"
    }
  ],
  "keyDefinitions": [
    {
      "term": "Difficult Term",
      "plainMeaning": "Easy 6-10 word explanation"
    }
  ],
  "answerStructure": "Suggested layout (e.g. 1 paragraph for comparison + 2 bullet points with evidence)."
}`;

      const response = await ai.models.generateContent({
        model: 'gemini-3.8-flash',
        contents: userContent,
        config: {
          systemInstruction,
          responseMimeType: 'application/json',
          responseSchema: {
            type: Type.OBJECT,
            properties: {
              cleanedPrompt: { type: Type.STRING },
              actionVerbs: {
                type: Type.ARRAY,
                items: { type: Type.STRING },
              },
              clearSteps: {
                type: Type.ARRAY,
                items: {
                  type: Type.OBJECT,
                  properties: {
                    stepNumber: { type: Type.INTEGER },
                    action: { type: Type.STRING },
                    hint: { type: Type.STRING },
                  },
                  required: ['stepNumber', 'action', 'hint'],
                },
              },
              keyDefinitions: {
                type: Type.ARRAY,
                items: {
                  type: Type.OBJECT,
                  properties: {
                    term: { type: Type.STRING },
                    plainMeaning: { type: Type.STRING },
                  },
                  required: ['term', 'plainMeaning'],
                },
              },
              answerStructure: { type: Type.STRING },
            },
            required: ['cleanedPrompt', 'actionVerbs', 'clearSteps', 'keyDefinitions', 'answerStructure'],
          },
        },
      });

      const data = JSON.parse(response.text?.trim() || 'null');
      const hasValidResult = Boolean(
        data &&
        typeof data.cleanedPrompt === 'string' &&
        Array.isArray(data.actionVerbs) &&
        Array.isArray(data.clearSteps) &&
        data.clearSteps.every((step: any) =>
          Number.isFinite(Number(step?.stepNumber)) &&
          typeof step?.action === 'string' &&
          typeof step?.hint === 'string',
        ) &&
        Array.isArray(data.keyDefinitions) &&
        data.keyDefinitions.every((definition: any) =>
          typeof definition?.term === 'string' &&
          typeof definition?.plainMeaning === 'string',
        ) &&
        typeof data.answerStructure === 'string',
      );

      if (hasValidResult) {
        res.json({ success: true, result: data });
        return;
      }

      throw new Error('Gemini returned an incomplete exam-helper response');
    }
  } catch (err) {
    console.error('Gemini exam-helper error:', err);
  }

  // High quality pedagogical fallback for exam helper
  const words = promptText.split(/\s+/);
  const verbs = ['Explain', 'Compare', 'Identify', 'Calculate', 'Describe', 'Analyze', 'Summarize', 'State'];
  const foundVerbs = verbs.filter(v => promptText.toLowerCase().includes(v.toLowerCase()));

  res.json({
    success: true,
    result: {
      cleanedPrompt: `State the main facts directly: 1) What happens, 2) Why it happens, and 3) Give one clear example.`,
      actionVerbs: foundVerbs.length > 0 ? foundVerbs : ['Explain', 'Identify'],
      clearSteps: [
        {
          stepNumber: 1,
          action: 'Read the question once and circle the main question word.',
          hint: 'Look for words like "How", "Why", or "Compare".',
        },
        {
          stepNumber: 2,
          action: 'State the core definition in your very first sentence.',
          hint: 'Do not repeat the entire question; jump straight to your answer.',
        },
        {
          stepNumber: 3,
          action: 'Provide one specific piece of evidence or classroom example.',
          hint: 'Name the specific organelle, event, or formula mentioned in class.',
        },
        {
          stepNumber: 4,
          action: 'Check that you addressed every part of the prompt before moving on.',
          hint: 'Count the sub-questions to ensure no free points were missed.',
        },
      ],
      keyDefinitions: [
        {
          term: 'Evaluate / Compare',
          plainMeaning: 'Show how two things are alike and how they differ.',
        },
        {
          term: 'Justify',
          plainMeaning: 'Give proof or reasons why your conclusion is right.',
        },
      ],
      answerStructure: 'Sentence 1: Direct claim. Sentence 2: Scientific reason. Sentence 3: Example from lesson.',
    },
  });
});

// 7. Teacher endpoints: mark question answered or post announcement
app.post('/api/rooms/:roomId/answer-question', (req: Request, res: Response) => {
  const room = getOrCreateRoom(req.params.roomId);
  const { questionId } = req.body;
  const q = room.questions.find(item => item.id === questionId);
  if (q) {
    q.answeredInClass = true;
    res.json({ success: true, question: q });
  } else {
    res.status(404).json({ error: 'Question not found' });
  }
});

app.post('/api/rooms/:roomId/notice', (req: Request, res: Response) => {
  const room = getOrCreateRoom(req.params.roomId);
  const { notice } = req.body;
  room.teacherNotice = notice;
  res.json({ success: true, notice });
});

// 8. Real-time Translation Endpoint (with Language Detection)
app.post('/api/rooms/:roomId/translate', async (req: Request, res: Response) => {
  const room = getOrCreateRoom(req.params.roomId);
  const { texts, targetLanguage } = req.body;

  if (!texts || !Array.isArray(texts) || texts.length === 0) {
    res.status(400).json({ error: 'Array of texts is required' });
    return;
  }

  const langMap: Record<string, string> = {
    es: 'Spanish',
    fr: 'French',
    zh: 'Mandarin Chinese (Simplified)',
    de: 'German',
    ja: 'Japanese',
    ar: 'Arabic',
    en: 'English',
  };

  const targetLangName = langMap[targetLanguage] || targetLanguage || 'Spanish';

  try {
    if (process.env.GEMINI_API_KEY) {
      const prompt = `You are a real-time educational translation engine for classroom captions and notes.
Detect the source language of these sentences and translate every single item accurately into ${targetLangName}.
Preserve pedagogical clarity, scientific/historical terms, and formatting.

Sentences to translate:
${JSON.stringify(texts)}

Return valid JSON with this schema:
{
  "detectedSourceLanguage": "Detected language name (e.g. English)",
  "translations": ["translated text 1", "translated text 2", ...]
}`;

      const response = await ai.models.generateContent({
        model: 'gemini-3.8-flash',
        contents: prompt,
        config: {
          responseMimeType: 'application/json',
          responseSchema: {
            type: Type.OBJECT,
            properties: {
              detectedSourceLanguage: { type: Type.STRING },
              translations: {
                type: Type.ARRAY,
                items: { type: Type.STRING },
              },
            },
            required: ['detectedSourceLanguage', 'translations'],
          },
        },
      });

      const parsed = JSON.parse(response.text?.trim() || '{}');
      if (parsed.translations && Array.isArray(parsed.translations)) {
        room.detectedLanguage = parsed.detectedSourceLanguage || 'English';
        res.json({
          success: true,
          detectedSourceLanguage: parsed.detectedSourceLanguage || 'English',
          translations: parsed.translations,
        });
        return;
      }
    }
  } catch (err) {
    console.error('Gemini translate error:', err);
  }

  // High-fidelity fallback translation dictionary for Spanish, French, Mandarin
  const dictionary: Record<string, Record<string, string>> = {
    es: {
      'Good morning everyone! Please settle down and open your science notebooks to unit 4.':
        '¡Buenos días a todos! Por favor, tomen asiento y abran sus cuadernos de ciencias en la unidad 4.',
      'Today we are answering a big question: How do living cells turn the food you eat into real energy to jump, think, and breathe?':
        'Hoy responderemos una gran pregunta: ¿Cómo convierten las células vivas los alimentos que comemos en energía real para saltar, pensar y respirar?',
      'In our last class we looked at chloroplasts capturing sunlight in plants. Today we meet their powerhouse partner: the mitochondrion.':
        'En la última clase vimos cómo los cloroplastos capturan la luz solar en las plantas. Hoy conoceremos a su socia energética: la mitocondria.',
      'Remember this chemical formula: Glucose plus oxygen yields carbon dioxide, water, and ATP energy.':
        'Recuerden esta fórmula química: Glucosa más oxígeno produce dióxido de carbono, agua y energía ATP.',
      'Think of ATP as a rechargeable microscopic battery. Every single muscle contraction uses up a tiny burst of ATP.':
        'Piensen en el ATP como una batería microscópica recargable. Cada contracción muscular consume una pequeña descarga de ATP.',
      'Without oxygen, cells can only produce a tiny trickle of energy through anaerobic fermentation, which is why your muscles burn when sprinting.':
        'Sin oxígeno, las células solo pueden producir un pequeño goteo de energía mediante fermentación anaeróbica, por eso los músculos queman al correr.',
      'Mitochondria Are the Powerhouses': 'Las mitocondrias son las centrales energéticas',
      'The Cellular Respiration Recipe': 'La receta de la respiración celular',
      'ATP Is the Rechargeable Battery': 'El ATP es la batería recargable',
      'Oxygen Makes It Efficient': 'El oxígeno lo hace eficiente',
    },
    fr: {
      'Good morning everyone! Please settle down and open your science notebooks to unit 4.':
        'Bonjour à tous ! Veuillez vous installer et ouvrir vos cahiers de sciences à l’unité 4.',
      'Today we are answering a big question: How do living cells turn the food you eat into real energy to jump, think, and breathe?':
        'Aujourd’hui, nous répondons à une grande question : comment les cellules vivantes transforment-elles la nourriture en énergie réelle pour sauter, penser et respirer ?',
      'In our last class we looked at chloroplasts capturing sunlight in plants. Today we meet their powerhouse partner: the mitochondrion.':
        'Lors du dernier cours, nous avons vu les chloroplastes capturer la lumière du soleil. Aujourd’hui, nous découvrons leur partenaire énergétique : la mitochondrie.',
      'Remember this chemical formula: Glucose plus oxygen yields carbon dioxide, water, and ATP energy.':
        'Rappelez-vous cette formule chimique : Glucose plus oxygène donne dioxyde de carbone, eau et énergie ATP.',
      'Think of ATP as a rechargeable microscopic battery. Every single muscle contraction uses up a tiny burst of ATP.':
        'Considérez l’ATP comme une batterie microscopique rechargeable. Chaque contraction musculaire consomme une micro-décharge d’ATP.',
      'Without oxygen, cells can only produce a tiny trickle of energy through anaerobic fermentation, which is why your muscles burn when sprinting.':
        'Sans oxygène, les cellules ne peuvent produire qu’un faible filet d’énergie par fermentation anaérobie, d’où la brûlure lors d’un sprint.',
      'Mitochondria Are the Powerhouses': 'Les mitochondries sont les centrales énergétiques',
      'The Cellular Respiration Recipe': 'La recette de la respiration cellulaire',
      'ATP Is the Rechargeable Battery': 'L’ATP est la batterie rechargeable',
      'Oxygen Makes It Efficient': 'L’oxygène la rend efficace',
    },
    zh: {
      'Good morning everyone! Please settle down and open your science notebooks to unit 4.':
        '大家早上好！请安静就座，翻开科学笔记本第4单元。',
      'Today we are answering a big question: How do living cells turn the food you eat into real energy to jump, think, and breathe?':
        '今天我们要探讨一个核心问题：活细胞是如何将摄入的食物转化为跳跃、思考和呼吸所需的真实能量的？',
      'In our last class we looked at chloroplasts capturing sunlight in plants. Today we meet their powerhouse partner: the mitochondrion.':
        '上节课我们学习了植物中捕获阳光的叶绿体。今天我们来认识它的能量伙伴：线粒体。',
      'Remember this chemical formula: Glucose plus oxygen yields carbon dioxide, water, and ATP energy.':
        '请记住这个化学反应式：葡萄糖加氧气生成二氧化碳、水和ATP能量。',
      'Think of ATP as a rechargeable microscopic battery. Every single muscle contraction uses up a tiny burst of ATP.':
        '可以把ATP看作微型可充电电池。每一次肌肉收缩都会消耗一小撮ATP能量。',
      'Without oxygen, cells can only produce a tiny trickle of energy through anaerobic fermentation, which is why your muscles burn when sprinting.':
        '没有氧气时，细胞只能通过无氧发酵产生微弱能量，这就是冲刺时肌肉酸痛的原因。',
      'Mitochondria Are the Powerhouses': '线粒体是细胞能量动力源',
      'The Cellular Respiration Recipe': '细胞呼吸反应配方',
      'ATP Is the Rechargeable Battery': 'ATP是可充电微电池',
      'Oxygen Makes It Efficient': '氧气带来极高转换效率',
    },
  };

  const targetDict = dictionary[targetLanguage] || {};
  const fallbackTranslations = texts.map(t => targetDict[t] || `[${targetLangName}] ${t}`);

  res.json({
    success: true,
    detectedSourceLanguage: 'English',
    translations: fallbackTranslations,
  });
});

// 9. Interactive Quiz Endpoints
// Create/save quiz
app.post('/api/rooms/:roomId/quiz/create', (req: Request, res: Response) => {
  const room = getOrCreateRoom(req.params.roomId);
  const { quiz } = req.body;

  if (!quiz || !quiz.title || !Array.isArray(quiz.questions)) {
    res.status(400).json({ error: 'Valid quiz data is required' });
    return;
  }

  const newQuiz: Quiz = {
    ...quiz,
    id: quiz.id || `quiz-${Date.now()}`,
    createdAt: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
    totalPoints: quiz.questions.reduce((sum: number, q: QuizQuestion) => sum + (q.points || 1), 0),
  };

  room.quizzes.unshift(newQuiz);
  res.json({ success: true, quiz: newQuiz });
});

// Generate quiz using Gemini 3.8 Flash from live classroom transcript
app.post('/api/rooms/:roomId/quiz/generate', async (req: Request, res: Response) => {
  const room = getOrCreateRoom(req.params.roomId);
  const transcriptText = room.captions.map(c => `${c.speaker}: ${c.text}`).join('\n');

  try {
    if (process.env.GEMINI_API_KEY && transcriptText.length > 20) {
      const prompt = `You are a curriculum assessment designer for ${room.subject} (${room.grade}, Topic: ${room.topic}).
Based on this classroom lesson transcript:
"""
${transcriptText}
"""

Create an accessible 4-question interactive check-in quiz for students.
Include:
- 3 Multiple Choice questions (with 4 distinct options, correctOptionId 'opt-1' through 'opt-4', clear explanation)
- 1 Short Answer question (with acceptableKeywords array, modelAnswer, explanation)

Return valid JSON conforming to this schema:
{
  "title": "Short Descriptive Title",
  "description": "Friendly description of what this checks",
  "questions": [
    {
      "id": "q-1",
      "type": "multiple_choice",
      "question": "Question text here?",
      "options": [
        { "id": "opt-1", "text": "Option 1" },
        { "id": "opt-2", "text": "Option 2" },
        { "id": "opt-3", "text": "Option 3" },
        { "id": "opt-4", "text": "Option 4" }
      ],
      "correctOptionId": "opt-1",
      "explanation": "Clear plain language explanation why this is right.",
      "points": 2
    },
    {
      "id": "q-2",
      "type": "short_answer",
      "question": "Short answer question here?",
      "acceptableKeywords": ["keyword1", "keyword2", "keyword3"],
      "modelAnswer": "Ideal 1-2 sentence student answer.",
      "explanation": "What grader looks for.",
      "points": 3
    }
  ]
}`;

      const response = await ai.models.generateContent({
        model: 'gemini-3.8-flash',
        contents: prompt,
        config: {
          responseMimeType: 'application/json',
          responseSchema: {
            type: Type.OBJECT,
            properties: {
              title: { type: Type.STRING },
              description: { type: Type.STRING },
              questions: {
                type: Type.ARRAY,
                items: {
                  type: Type.OBJECT,
                  properties: {
                    id: { type: Type.STRING },
                    type: { type: Type.STRING },
                    question: { type: Type.STRING },
                    options: {
                      type: Type.ARRAY,
                      items: {
                        type: Type.OBJECT,
                        properties: {
                          id: { type: Type.STRING },
                          text: { type: Type.STRING },
                        },
                        required: ['id', 'text'],
                      },
                    },
                    correctOptionId: { type: Type.STRING },
                    acceptableKeywords: {
                      type: Type.ARRAY,
                      items: { type: Type.STRING },
                    },
                    modelAnswer: { type: Type.STRING },
                    explanation: { type: Type.STRING },
                    points: { type: Type.INTEGER },
                  },
                  required: ['id', 'type', 'question', 'explanation', 'points'],
                },
              },
            },
            required: ['title', 'description', 'questions'],
          },
        },
      });

      const parsed = JSON.parse(response.text?.trim() || '{}');
      if (parsed.questions && Array.isArray(parsed.questions)) {
        const generatedQuiz: Quiz = {
          id: `quiz-gen-${Date.now()}`,
          title: parsed.title || `${room.topic} Quick Check`,
          description: parsed.description || 'Test your knowledge on key points from today’s lecture.',
          questions: parsed.questions,
          createdAt: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
          status: 'active',
          totalPoints: parsed.questions.reduce((sum: number, q: any) => sum + (q.points || 2), 0),
        };

        room.quizzes.unshift(generatedQuiz);
        res.json({ success: true, quiz: generatedQuiz });
        return;
      }
    }
  } catch (err) {
    console.error('Gemini quiz generate error:', err);
  }

  // Fallback generation if no API key
  const fallbackQuiz: Quiz = {
    id: `quiz-gen-${Date.now()}`,
    title: `${room.topic} Understanding Check`,
    description: 'Quick check-in generated from the core lesson concepts.',
    createdAt: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
    status: 'active',
    totalPoints: 8,
    questions: [
      {
        id: `q-gen-1`,
        type: 'multiple_choice',
        question: `What is the main function discussed during today's lesson on ${room.topic}?`,
        options: [
          { id: 'opt-1', text: 'Producing and managing energy conversions cleanly' },
          { id: 'opt-2', text: 'Destroying all active cellular membranes' },
          { id: 'opt-3', text: 'Freezing metabolic activity indefinitely' },
          { id: 'opt-4', text: 'Stopping respiratory reactions' },
        ],
        correctOptionId: 'opt-1',
        explanation: 'The primary concept focuses on how systems sustain balanced and efficient energy flows.',
        points: 2,
      },
      {
        id: `q-gen-2`,
        type: 'short_answer',
        question: `State one key reason why this topic is essential for living systems.`,
        acceptableKeywords: ['energy', 'fuel', 'life', 'survive', 'cell', 'function', 'atp', 'power'],
        modelAnswer: 'It provides the continuous energy required for cellular work, growth, and survival.',
        explanation: 'A strong answer connects the chemical mechanism directly to biological function.',
        points: 3,
      },
      {
        id: `q-gen-3`,
        type: 'multiple_choice',
        question: `Which factor most significantly increases the efficiency of this process?`,
        options: [
          { id: 'opt-1', text: 'Presence of oxygen and adequate surface area' },
          { id: 'opt-2', text: 'Complete absence of nutrients' },
          { id: 'opt-3', text: 'Total darkness and extreme cold' },
          { id: 'opt-4', text: 'Blocking all membrane receptors' },
        ],
        correctOptionId: 'opt-1',
        explanation: 'Aerobic pathways and folded membranes maximize chemical contact and ATP output.',
        points: 3,
      },
    ],
  };

  room.quizzes.unshift(fallbackQuiz);
  res.json({ success: true, quiz: fallbackQuiz });
});

// Update quiz status (active, draft, closed)
app.post('/api/rooms/:roomId/quiz/:quizId/status', (req: Request, res: Response) => {
  const room = getOrCreateRoom(req.params.roomId);
  const { status } = req.body;
  const quiz = room.quizzes.find(q => q.id === req.params.quizId);
  if (quiz && (status === 'active' || status === 'draft' || status === 'closed')) {
    quiz.status = status;
    res.json({ success: true, quiz });
  } else {
    res.status(404).json({ error: 'Quiz not found' });
  }
});

// Submit quiz answers & grade
app.post('/api/rooms/:roomId/quiz/:quizId/submit', async (req: Request, res: Response) => {
  const room = getOrCreateRoom(req.params.roomId);
  const quiz = room.quizzes.find(q => q.id === req.params.quizId);

  if (!quiz) {
    res.status(404).json({ error: 'Quiz not found' });
    return;
  }

  const { studentName = 'Quiet Student', answers = {}, sharedWithTeacher = true } = req.body;

  let totalScore = 0;
  let maxScore = quiz.totalPoints;
  const questionResults: Record<string, { isCorrect: boolean; earnedPoints: number; feedback: string }> = {};

  for (const q of quiz.questions) {
    const studentAns = (answers[q.id] || '').trim();

    if (q.type === 'multiple_choice') {
      const isCorrect = studentAns === q.correctOptionId;
      const earned = isCorrect ? q.points : 0;
      totalScore += earned;
      questionResults[q.id] = {
        isCorrect,
        earnedPoints: earned,
        feedback: isCorrect
          ? `Correct! (+${earned} pts) — ${q.explanation}`
          : `Not quite. (+0 pts) — Correct answer was: ${q.options?.find(o => o.id === q.correctOptionId)?.text}. ${q.explanation}`,
      };
    } else {
      // Short answer grading
      const lower = studentAns.toLowerCase();
      const keywords = q.acceptableKeywords || [];
      const matchCount = keywords.filter(k => lower.includes(k.toLowerCase())).length;

      let isCorrect = false;
      let earned = 0;

      if (matchCount >= 2 || (keywords.length <= 1 && matchCount >= 1) || studentAns.length > 30) {
        isCorrect = true;
        earned = q.points;
      } else if (matchCount === 1) {
        isCorrect = true;
        earned = Math.max(1, q.points - 1);
      }

      totalScore += earned;
      questionResults[q.id] = {
        isCorrect,
        earnedPoints: earned,
        feedback: isCorrect
          ? `Great answer! (+${earned}/${q.points} pts) — ${q.explanation}`
          : `Good effort! (+${earned}/${q.points} pts) — Model answer: "${q.modelAnswer}". ${q.explanation}`,
      };
    }
  }

  const percentage = Math.round((totalScore / Math.max(1, maxScore)) * 100);

  const submission: StudentQuizSubmission = {
    id: `sub-${Date.now()}-${Math.random().toString(36).slice(2, 6)}`,
    studentName,
    quizId: quiz.id,
    score: totalScore,
    maxScore,
    percentage,
    answers,
    questionResults,
    submittedAt: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
    sharedWithTeacher: Boolean(sharedWithTeacher),
  };

  if (submission.sharedWithTeacher) {
    room.quizSubmissions.unshift(submission);
  }

  res.json({ success: true, submission });
});

// Get quiz submissions (for teacher)
app.get('/api/rooms/:roomId/quiz/:quizId/submissions', (req: Request, res: Response) => {
  const room = getOrCreateRoom(req.params.roomId);
  const subs = room.quizSubmissions.filter(s => s.quizId === req.params.quizId);
  res.json({ success: true, submissions: subs });
});

// Production static file serving & Vite development middleware
if (process.env.NODE_ENV === 'production') {
  app.use(express.static(path.join(__dirname, 'dist')));
  app.get('*', (_req: Request, res: Response) => {
    res.sendFile(path.join(__dirname, 'dist', 'index.html'));
  });
} else {
  // Mount Vite middlewares in development
  import('vite').then(async ({ createServer }) => {
    const vite = await createServer({
      server: { middlewareMode: true },
      appType: 'spa',
    });
    app.use(vite.middlewares);
  });
}

app.listen(PORT, '0.0.0.0', () => {
  console.log(`ClassEase server running on http://0.0.0.0:${PORT}`);
});
