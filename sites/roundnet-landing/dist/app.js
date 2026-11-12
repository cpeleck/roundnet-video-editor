document.documentElement.classList.add('js');

const clips = [
  { name: 'Rally 01 · Ace', time: '00:14 – 00:24', score: [7, 6], duration: 10, starred: false },
  { name: 'Rally 02 · Sideout', time: '00:36 – 00:44', score: [8, 6], duration: 8, starred: true },
  { name: 'Rally 03 · Defensive break', time: '01:14 – 01:23', score: [8, 7], duration: 9, starred: false },
  { name: 'Rally 04 · Sideout', time: '01:49 – 02:04', score: [9, 7], duration: 15, starred: true },
];
let mode = 'match';
let selected = 1;
const modeButtons = [...document.querySelectorAll('[data-mode]')];
const clipButtons = [...document.querySelectorAll('[data-clip]')];

function renderPreview() {
  const visible = clips.map((clip, index) => ({ ...clip, index })).filter(clip => mode === 'match' || clip.starred);
  if (!visible.some(clip => clip.index === selected)) selected = visible[0].index;
  modeButtons.forEach(button => button.setAttribute('aria-pressed', String(button.dataset.mode === mode)));
  clipButtons.forEach(button => {
    const index = Number(button.dataset.clip);
    button.hidden = !visible.some(clip => clip.index === index);
    button.classList.toggle('is-selected', selected === index);
    button.setAttribute('aria-pressed', String(selected === index));
  });
  const clip = clips[selected];
  document.querySelector('#sample-name').textContent = clip.name;
  document.querySelector('#sample-time').textContent = clip.time;
  const score = document.querySelector('#demo-score');
  const dash = document.createElement('i');
  dash.textContent = '–';
  score.replaceChildren(document.createTextNode(String(clip.score[0]).padStart(2, '0') + ' '), dash, document.createTextNode(' ' + String(clip.score[1]).padStart(2, '0')));
  document.querySelector('#clip-summary').textContent = `${visible.length} clips · ${visible.reduce((total, clip) => total + clip.duration, 0)} seconds`;
}

modeButtons.forEach(button => button.addEventListener('click', () => { mode = button.dataset.mode; renderPreview(); }));
clipButtons.forEach(button => button.addEventListener('click', () => { selected = Number(button.dataset.clip); renderPreview(); }));

const menuButton = document.querySelector('.menu-toggle');
const navigation = document.querySelector('#site-navigation');
function closeMenu() {
  menuButton.setAttribute('aria-expanded', 'false');
  menuButton.setAttribute('aria-label', 'Open navigation');
  navigation.classList.remove('is-open');
}
menuButton.addEventListener('click', () => {
  const open = menuButton.getAttribute('aria-expanded') !== 'true';
  menuButton.setAttribute('aria-expanded', String(open));
  menuButton.setAttribute('aria-label', open ? 'Close navigation' : 'Open navigation');
  navigation.classList.toggle('is-open', open);
});
navigation.querySelectorAll('a').forEach(link => link.addEventListener('click', closeMenu));
document.addEventListener('keydown', event => {
  if (event.key === 'Escape' && menuButton.getAttribute('aria-expanded') === 'true') { closeMenu(); menuButton.focus(); }
});
document.addEventListener('click', event => { if (!event.target.closest('.site-header')) closeMenu(); });
window.matchMedia('(min-width: 541px)').addEventListener('change', closeMenu);
renderPreview();
