import React from 'react';
import { Link } from 'react-router-dom';
import { motion } from 'framer-motion';

const fadeUp = {
  hidden:  { opacity: 0, y: 24, filter: 'blur(4px)' },
  visible: (d = 0) => ({
    opacity: 1, y: 0, filter: 'blur(0px)',
    transition: { duration: 0.72, delay: d, ease: [0.16, 1, 0.3, 1] }
  })
};

/** Animate child as it enters the viewport. */
export function FadeUp({ children, delay = 0, style, className }) {
  return (
    <motion.div
      variants={fadeUp}
      initial="hidden"
      whileInView="visible"
      viewport={{ once: true, margin: '-50px' }}
      custom={delay}
      style={style}
      className={className}
    >
      {children}
    </motion.div>
  );
}

/** Warp-style page transition wrapper. */
export function PageTransition({ children }) {
  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.985, filter: 'blur(8px)' }}
      animate={{ opacity: 1, scale: 1,     filter: 'blur(0px)' }}
      exit={{    opacity: 0, scale: 1.015, filter: 'blur(8px)' }}
      transition={{ duration: 0.52, ease: [0.16, 1, 0.3, 1] }}
    >
      {children}
    </motion.div>
  );
}
