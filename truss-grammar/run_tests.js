// Usage: node run_tests.js
const { runTests } = require('./core.js');
const results = runTests();
results.forEach(t => console.log((t.info ? 'INFO' : t.pass ? 'PASS' : 'FAIL') + '  ' + t.name + '\n      ' + t.detail));
const real = results.filter(t => !t.info);
console.log('\n' + real.filter(t => t.pass).length + ' of ' + real.length + ' checks pass');
process.exit(real.every(t => t.pass) ? 0 : 1);
