import {Slot} from '@radix-ui/react-slot'; import {cva} from 'class-variance-authority'; import {clsx} from 'clsx'; import {twMerge} from 'tailwind-merge'; import type {ComponentProps} from 'react';
export const cn=(...v:Parameters<typeof clsx>)=>twMerge(clsx(...v));
const variants=cva('button',{variants:{variant:{default:'primary',outline:'outline',ghost:'ghost'}},defaultVariants:{variant:'default'}});
export function Button({className,variant,asChild,...props}:ComponentProps<'button'>&{variant?:'default'|'outline'|'ghost';asChild?:boolean}){const C=asChild?Slot:'button';return <C className={cn(variants({variant}),className)} {...props}/>}
export function Card({className,...props}:ComponentProps<'section'>){return <section className={cn('panel',className)} {...props}/>}
